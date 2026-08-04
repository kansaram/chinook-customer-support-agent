# app.py
import gradio as gr

# Build your Gradio UI block...
with gr.Blocks() as demo:
    gr.Markdown("# Chinook Customer Support Agent")
    # ... your chat UI components ...

if __name__ == "__main__":
    # Ensure server_name="0.0.0.0" so it binds to all network interfaces inside Docker
    demo.launch(server_name="0.0.0.0", server_port=7860)