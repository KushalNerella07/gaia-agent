# agent.py – GAIA-ready LangGraph agent (LM Studio or HF endpoint, chat→raw API)
import os, time, json, re
from dotenv import load_dotenv

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

# keep LangChain only for messages, tools, graph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader

# raw OpenAI client (LM Studio is OpenAI-compatible)
from openai import OpenAI

load_dotenv()

# ────────────── TOOLS ──────────────
@tool
def multiply(a: int, b: int) -> int:
    """Return a × b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Return a + b."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Return a − b."""
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Return a ÷ b. Raises ValueError if b == 0."""
    if b == 0:
        raise ValueError("divide-by-zero")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Return a mod b."""
    return a % b

@tool
def wiki_search(query: str) -> str:
    """Return the two most relevant Wikipedia pages for *query*."""
    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return "\n\n---\n\n".join(d.page_content for d in docs)

@tool
def arxiv_search(query: str) -> str:
    """Return the first 1 000 chars of the top-3 arXiv papers for *query*."""
    docs = ArxivLoader(query=query, load_max_docs=3).load()
    return "\n\n---\n\n".join(d.page_content[:1000] for d in docs)

TOOLS = [multiply, add, subtract, divide, modulus, wiki_search, arxiv_search]

# ─────────── SYSTEM PROMPT (text only) ───────────
SYS_TXT = open("system_prompt.txt", encoding="utf-8").read()

# ─────────── LM STUDIO CLIENT ───────────
LMSTUDIO_BASE = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_KEY  = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
# **Use the exact API identifier shown in LM Studio’s right panel**
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "mistral-7b-instruct-v0.2")

client = OpenAI(base_url=LMSTUDIO_BASE, api_key=LMSTUDIO_KEY)

# ─────────── TOOL-CALL PARSER ─────────
CALL_RE = re.compile(r"^CALL_TOOL\s+(\w+)(?:\s+(.*))?$", re.I)

def parse_call(text: str):
    m = CALL_RE.match(text.strip())
    if not m:
        return None, None
    name, arg_json = m.group(1), m.group(2) or "{}"
    try:
        args = json.loads(arg_json)
    except json.JSONDecodeError:
        args = {}
    return name, args

# ─────────── Pack whole history → single prompt ─────────
def pack_prompt(history):
    lines = []
    for m in history:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"ASSISTANT: {m.content}")
        else:
            # tool messages or other
            content = getattr(m, "content", "")
            name = getattr(m, "name", "TOOL")
            lines.append(f"{name.upper()}: {content}")
    convo = "\n".join(lines)
    # The agent instructions + entire conversation as plain text
    return f"{SYS_TXT}\n\n---\nConversation so far:\n{convo}\n\nASSISTANT:"

# ─────────── GRAPH NODES ──────────────
def assistant(state: MessagesState):
    prompt = pack_prompt(state["messages"])

    # Try chat endpoint first with exactly ONE user message.
    # If LM Studio’s template still complains, fall back to /v1/completions.
    for attempt in range(3):
        try:
            # 1) preferred: chat.completions with one {role:user} message
            resp = client.chat.completions.create(
                model=LMSTUDIO_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e1:
            # 2) fallback: classic completions (no roles → bypasses jinja)
            try:
                resp2 = client.completions.create(
                    model=LMSTUDIO_MODEL,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=512,
                )
                text = (resp2.choices[0].text or "").strip()
            except Exception as e2:
                print(f"⚠️  LLM call failed (try {attempt+1}/3): {e1 or e2}")
                time.sleep(2)
                continue

        # tool request?
        tool, args = parse_call(text)
        if tool:
            state["messages"].append(AIMessage(content=text, tool=tool, args=args))
            return {"messages": state["messages"]}

        # normal reply
        state["messages"].append(AIMessage(content=text))
        return {"messages": state["messages"]}

    raise RuntimeError("LLM failed after 3 attempts")

def build_graph():
    g = StateGraph(MessagesState)
    g.add_node("assistant", assistant)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "assistant")
    g.add_conditional_edges("assistant", tools_condition)
    g.add_edge("tools", "assistant")
    return g.compile()

class BasicAgent:
    def __init__(self):
        print("🔧  Initialising BasicAgent …")
        self.graph = build_graph()
    def __call__(self, q: str) -> str:
        out = self.graph.invoke({"messages": [HumanMessage(content=q)]})
        return out["messages"][-1].content.strip()

if __name__ == "__main__":
    print("2 + 3 =", BasicAgent()("What is 2 + 3?"))
