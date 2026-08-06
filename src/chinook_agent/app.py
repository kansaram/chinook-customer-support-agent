import atexit
import gradio as gr
import os
import signal
import sys
from uuid import uuid4

from chinook_agent.graph import build_graph

compiled_graph = build_graph()

def handle_message(user_message: str, history, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = compiled_graph.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )

    if result.get("response"):
        return result["response"]

    messages = result.get("messages", [])
    if not messages:
        return "I couldn't generate a response."

    last_message = messages[-1]
    return getattr(last_message, "content", "") or "I couldn't generate a response."


def create_thread_id() -> str:
    return str(uuid4())


def close_demo() -> None:
    try:
        demo.close()
    except Exception:
        pass


def handle_shutdown(_signum=None, _frame=None) -> None:
    close_demo()
    raise SystemExit(0)

with gr.Blocks() as demo:
    gr.Markdown("# Chinook Customer Support Agent")
    gr.Markdown("Ask about invoices, tracks, artists, genres, and music preferences.")

    session_thread_id = gr.State(create_thread_id())
    chatbot = gr.Chatbot(label="Support Chat")
    message = gr.Textbox(label="Message", placeholder="Ask about artists, tracks, invoices, or preferences...")
    clear = gr.Button("Clear")

    def submit_message(user_message: str, history: list, thread_id: str):
        reply = handle_message(user_message, history, thread_id)
        updated_history = (history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]
        return updated_history, ""

    message.submit(
        submit_message,
        inputs=[message, chatbot, session_thread_id],
        outputs=[chatbot, message],
    )

    clear.click(
        lambda: ([], "", create_thread_id()),
        outputs=[chatbot, message, session_thread_id],
    )

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    atexit.register(close_demo)
    signal.signal(signal.SIGINT, handle_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        demo.launch(server_name=server_name, server_port=server_port, share=True)
    finally:
        close_demo()