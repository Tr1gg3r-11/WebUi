from ..extras.packages import is_gradio_available
if is_gradio_available():
    import gradio as gr
from ..extras.error import validate_value
def download(name: str) -> None:
    print(f"下载{name}数据集")

def convert(load_dir: str, save_dir: str, tokenizer_path: str, workers: int) -> None:
    if validate_value(workers,'多线程处理数'):
        print(f"{workers}线程利用{tokenizer_path}的tokenizer-type将{load_dir}数据集处理到{save_dir}")