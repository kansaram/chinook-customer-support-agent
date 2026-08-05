# app.py
import gradio as gr

from chinook_agent.graph import build_graph

compiled_graph = build_graph()

def handle_message(user_message: str, history, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = compiled_graph.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )
    return result["messages"][-1].content

# Build your Gradio UI block...
with gr.Blocks() as demo:
    gr.Markdown("# Chinook Customer Support Agent")
    # ... your chat UI components ...

if __name__ == "__main__":
    # Ensure server_name="0.0.0.0" so it binds to all network interfaces inside Docker
    demo.launch(server_name="0.0.0.0", server_port=7860)