from .webui.interface import run_web_ui
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio.routes")
if __name__ == "__main__":
    run_web_ui()