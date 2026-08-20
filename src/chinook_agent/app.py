import atexit
import gradio as gr
import os
import signal
import sys
from uuid import uuid4

from chinook_agent.graph import build_graph
from chinook_agent.database.health import health_check
from chinook_agent.config.logging import get_logger

logger = get_logger(__name__)
compiled_graph = build_graph()


from langgraph.errors import GraphRecursionError

# Caps total graph steps per turn. The runaway loop we saw (memory_agent calling
# get_preferences repeatedly) reached 14+ steps and was still going — this limit
# stops ANY loop, in ANY agent/tool, current or future, without needing to patch
# every individual tool. A legitimate multi-agent turn (handoff + tool calls)
# typically takes well under 10 steps, so 20 gives real headroom without letting
# a runaway loop burn more than a few seconds/cents before being cut off.
GRAPH_RECURSION_LIMIT = 20


def handle_message(user_message: str, history, thread_id: str):
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }
    try:
        result = compiled_graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
        )
    except GraphRecursionError:
        logger.error(f"Graph exceeded recursion limit ({GRAPH_RECURSION_LIMIT} steps) — likely a loop", exc_info=True)
        return (
            "Sorry, I got stuck trying to process that request. "
            "Could you try rephrasing it, or asking one thing at a time?"
        )

    parts = result.get("response_parts")
    if parts:
        return "\n\n".join(parts)

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

    session_thread_id = gr.State(create_thread_id)

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
    # Startup health check — fail fast with a clear message rather than launching
    # and having every conversation break confusingly the moment a DB is touched.
    print("Checking database connectivity...")
    status = health_check()
    if not status["healthy"]:
        print("STARTUP FAILED — one or more databases are not reachable:")
        print(f"  chinook.db: {status['chinook_db']}")
        print(f"  customer_memory.db: {status['memory_db']}")
        sys.exit(1)
    print("Database health check passed.")

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