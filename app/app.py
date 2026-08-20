import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from formatting import parse_vlm_output

VLM_ENGINE = None

def init_engine():
    global VLM_ENGINE
    if VLM_ENGINE is not None:
        return True
        
    try:
        from src.vlm.inference import VLMInferenceEngine
        VLM_ENGINE = VLMInferenceEngine("configs/model.yaml")
        VLM_ENGINE.load()
        return True
    except Exception as e:
        print(f"Warning: Could not initialize VLM Engine: {e}")
        return False

def analyze_meme(image_path, cultural_mode):
    if image_path is None:
        return "Error", "N/A", "N/A", "N/A", "N/A", "Please upload a valid image."
        
    if not init_engine():
        return "Error", "N/A", "N/A", "N/A", "N/A", "Model is currently unavailable or missing dependencies locally."
        
    try:
        result = VLM_ENGINE.infer(image_path)
        return parse_vlm_output(result)
    except Exception as e:
        return "Error", "N/A", "N/A", "N/A", "N/A", f"An internal inference error occurred: {str(e)}"

def main():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Please run: uv pip install gradio")
        sys.exit(0)
        
    from ui_components import create_ui
    demo = create_ui(analyze_meme)
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)

if __name__ == "__main__":
    print("Gradio app entrypoint initialized.")
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-test":
        sys.exit(0)
    else:
        main()
