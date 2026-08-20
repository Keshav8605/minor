import sys
try:
    import gradio as gr
except ImportError:
    gr = None

def create_ui(analyze_fn):
    if gr is None:
        print("Gradio is not installed. Cannot create UI.")
        sys.exit(1)
        
    with gr.Blocks(title="Culturally Aware Humor Detection", theme=gr.themes.Default()) as demo:
        gr.Markdown("# Culturally Aware Multimodal Humor Detection in Hindi")
        gr.Markdown("Upload a codemixed Hindi/English meme to see how explicit cultural context changes the VLM's reasoning.")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Meme")
                cultural_toggle = gr.Radio(
                    choices=["Off", "On"], 
                    value="Off", 
                    label="Use Cultural Context (Mode A vs Mode B)"
                )
                analyze_btn = gr.Button("Analyze Meme", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### Results")
                pred_out = gr.Textbox(label="Humor Prediction", interactive=False)
                conf_out = gr.HTML(label="Confidence")
                text_out = gr.Textbox(label="Detected Text (OCR)", interactive=False)
                
                gr.Markdown("### Cultural Analysis")
                cat_out = gr.Textbox(label="Cultural Category", interactive=False)
                dep_out = gr.Textbox(label="Cultural Dependency", interactive=False)
                reason_out = gr.Textbox(label="Reasoning", interactive=False, lines=4)
                
        analyze_btn.click(
            fn=analyze_fn,
            inputs=[image_input, cultural_toggle],
            outputs=[pred_out, conf_out, text_out, cat_out, dep_out, reason_out]
        )
        
    return demo
