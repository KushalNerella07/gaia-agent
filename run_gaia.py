# run_gaia.py
import os, re, json, logging, requests, sys
from pathlib import Path
from dotenv import load_dotenv
from agent import build_graph
from langchain_core.messages import HumanMessage

# ---- small helpers (same behavior as gaia_format.py) --------------------
def extract_final(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r"final\s*answer\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()
    return str(text).strip()

def canonical(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"[\r\n\t]+", " ", text)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()

# ---- logging -------------------------------------------------------------
log = logging.getLogger("gaia")
log.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
log.addHandler(h)

# ---- env / config --------------------------------------------------------
load_dotenv()
API         = "https://agents-course-unit4-scoring.hf.space"
USERNAME    = os.getenv("GAIA_USERNAME", "KushCodes")
AGENT_LINK  = os.getenv("GAIA_AGENT_LINK", "https://huggingface.co/spaces/KushCodes/unit4-gaia-agent")

OUT_RAW     = Path("submission_raw.jsonl")
OUT_CLEAN   = Path("submission.jsonl")  # this is the one we submit

# ---- normalization -------------------------------------------------------
NUM_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
    "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20
}
NUM_QUESTION_HINTS = (
    "how many","number of","count","total","sum",
    "pages","at bats","albums","year","distance",
    "price","age","length","duration","score","wins","goals","population"
)

def _strip_wrapping(text: str) -> str:
    t = text
    if "(" in t:
        t = t.split("(", 1)[0].strip()
    t = re.sub(r'^\s*["\'\[\(]+', "", t)
    t = re.sub(r'["\'\]\)]+\s*$', "", t).strip()
    t = t.rstrip(" .").strip()
    return t

def _prefer_count_number(s: str) -> str | None:
    nums = [m.group(0) for m in re.finditer(r"-?\d+(?:\.\d+)?", s.replace(",", ""))]
    if not nums:
        return None
    def looks_like_year(x: str) -> bool:
        if re.fullmatch(r"\d{4}", x):
            v = int(x)
            return 1500 <= v <= 2100
        return False
    candidates = [n for n in nums if not looks_like_year(n)] or nums[:]
    def key(n: str):
        is_int = re.fullmatch(r"-?\d+", n) is not None
        v = float(n)
        return (0 if is_int else 1, abs(v), v)
    candidates.sort(key=key)
    best = candidates[0]
    if re.fullmatch(r"-?\d+\.0", best):
        best = best[:-2]
    return best

def normalize_answer(raw_text: str, question: str, lowercase_lists: bool = True) -> str:
    extracted = extract_final(raw_text)
    s = canonical(extracted).strip()

    if s.lower().startswith("final answer:"):
        s = s[len("final answer:"):].strip()

    if re.search(r"<[^>]+>", s):
        return "Unknown"

    s = _strip_wrapping(s) or "Unknown"

    if s.lower().startswith("unknown"):
        return "Unknown"

    # MCQ: a-e letters only
    mcq = s.replace(" ", "").lower()
    if re.fullmatch(r"[a-e](,[a-e])*", mcq or ""):
        parts = sorted(set(mcq.split(",")))
        return ",".join(parts)

    # numeric-leaning questions
    ql = (question or "").lower()
    if any(h in ql for h in NUM_QUESTION_HINTS):
        best = _prefer_count_number(s)
        if best is not None:
            return best
        for w, n in NUM_WORDS.items():
            if re.search(rf"\b{w}\b", s.lower()):
                return str(n)
        return "Unknown"

    # lists
    if "," in s:
        items = [i.strip() for i in s.split(",") if i.strip()]
        if lowercase_lists:
            items = [i.lower() for i in items]
        return ",".join(items) if items else "Unknown"

    # default
    return s

# ---- main flow -----------------------------------------------------------
def main():
    log.info("⚙️  Building agent …")
    graph = build_graph()

    # fetch tasks
    log.info("📥  Fetching questions …")
    try:
        resp = requests.get(f"{API}/questions", timeout=30)
        resp.raise_for_status()
        tasks = resp.json()
        if not tasks:
            log.error("No tasks received from API.")
            sys.exit(1)
        log.info("✅ Got %d tasks", len(tasks))
    except Exception as e:
        log.error("Failed to fetch questions: %s", e)
        sys.exit(1)

    # run agent -> raw answers
    raw_recs = []
    for i, task in enumerate(tasks, 1):
        tid = task.get("task_id")
        q = task.get("question", "")
        log.info("[ %2d/%d] %s → Running agent …", i, len(tasks), tid)
        try:
            out = graph.invoke({"messages": [HumanMessage(content=q)]})
            raw = (out["messages"][-1].content or "").strip()
        except Exception as e:
            raw = "FINAL ANSWER: Unknown"
            log.error("Agent error on %s: %s", tid, e)
        raw_recs.append({"task_id": tid, "submitted_answer": raw})

    # write RAW file for debugging
    OUT_RAW.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in raw_recs), encoding="utf-8")
    log.info("💾 Wrote raw answers: %s", OUT_RAW)

    # normalize -> cleaned answers
    cleaned = []
    for r in raw_recs:
        tid = r["task_id"]
        q = next((t["question"] for t in tasks if t["task_id"] == tid), "")
        raw = r["submitted_answer"]
        norm = normalize_answer(raw, q, lowercase_lists=True)

        raw_extract = extract_final(raw).strip()
        if norm != raw_extract:
            log.info("⇢ %s\n    RAW : %s\n    NORM: %s", tid, raw_extract, norm)

        cleaned.append({"task_id": tid, "submitted_answer": norm})

    # write CLEAN file (submitted)
    OUT_CLEAN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in cleaned), encoding="utf-8")
    log.info("💾 Wrote cleaned answers: %s", OUT_CLEAN)

    # submit CLEAN
    log.info("🚀 Submitting your answers …")
    payload = {
        "username": USERNAME.strip(),
        "agent_code": AGENT_LINK,
        "answers": cleaned,
    }
    try:
        sresp = requests.post(f"{API}/submit", json=payload, timeout=60)
        sresp.raise_for_status()
        log.info("📬 Submission response: %s", sresp.json())
    except Exception as e:
        log.error("Submission failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
