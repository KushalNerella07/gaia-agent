""" Basic Agent Evaluation Runner """
import os
import gradio as gr
import requests
import pandas as pd
from langchain_core.messages import HumanMessage
from agent import build_graph
from langchain_core.tools import tool


# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
class BasicAgent:
    """A LangGraph agent."""
    def __init__(self):
        print("BasicAgent initialized.")
        self.graph = build_graph()

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        messages = [HumanMessage(content=question)]
        messages = self.graph.invoke({"messages": messages})
        answer = messages['messages'][-1].content
        return answer[14:]  # assumes all answers are prefixed with 'FINAL ANSWER: '

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Fetches all questions, runs the agent, submits all answers, and shows the result."""
    space_id = os.getenv("SPACE_ID")
    if profile:
        username = profile.username
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please log in to Hugging Face with the button.", None

    questions_url = f"{DEFAULT_API_URL}/questions"
    submit_url = f"{DEFAULT_API_URL}/submit"

    try:
        agent = BasicAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(f"Agent code: {agent_code}")

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty.", None
    except Exception as e:
        return f"Error fetching questions: {e}", None

    results_log = []
    answers_payload = []

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or not question_text:
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": f"AGENT ERROR: {e}"
            })

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results_log)

    except requests.exceptions.HTTPError as e:
        error_detail = f"HTTP Error: {e.response.status_code} - {e.response.text}"
        return f"Submission Failed: {error_detail}", pd.DataFrame(results_log)
    except requests.exceptions.RequestException as e:
        return f"Submission Failed: Network error - {e}", pd.DataFrame(results_log)
    except Exception as e:
        return f"Unexpected error during submission: {e}", pd.DataFrame(results_log)

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Basic Agent Evaluation Runner")
    gr.Markdown("""
    ### Instructions:
    1. Clone this Space and customize your agent logic.
    2. Log in to your Hugging Face account below.
    3. Click 'Run Evaluation & Submit All Answers' to run the agent and submit.
    
    ---
    **Note:** Submission may take a while as it processes all questions. This is a minimal setup. You're encouraged to optimize!
    """)

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Status", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")

    if space_host:
        print(f"✅ SPACE_HOST: {space_host}")
        print(f"   URL: https://{space_host}.hf.space")
    else:
        print("ℹ️  SPACE_HOST not set. Running locally?")

    if space_id:
        print(f"✅ SPACE_ID: {space_id}")
        print(f"   Repo: https://huggingface.co/spaces/{space_id}")
    else:
        print("ℹ️  SPACE_ID not found. Cannot generate repo link.")

    print("Launching Gradio App...\n")
    demo.launch(debug=True, share=False)
