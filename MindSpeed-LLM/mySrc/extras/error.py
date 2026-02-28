from .packages import is_gradio_available
if is_gradio_available():
    import gradio as gr
from .constants import MINIMUM, MAXIMUM
def validate_value(value : enumerate[float, int], label : str) -> bool:
    try:
        if label in MINIMUM and value < MINIMUM[label]:
            gr.Warning(f"⚠ <strong>{label}</strong>的值不可小于{MINIMUM[label]},请输入正确的值!")
            return False
        if label in MAXIMUM and value > MAXIMUM[label]:
            gr.Warning(f"⚠ <strong>{label}</strong>的值不可大于{MAXIMUM[label]},请输入正确的值!")
            return False
    except (ValueError, TypeError) as e:
        gr.Warning(e)
        return False
    return True