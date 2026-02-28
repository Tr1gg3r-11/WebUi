from ..extras.packages import is_gradio_available
if is_gradio_available():
    import gradio as gr
from ..extras.error import validate_value
from ..extras.constants import DATASETS_SH, DATA_CONVERT_SH
import subprocess
import os
def download(name: str, source: str, download_dir: str) -> None:
    choice = name + '-' + source
    file_path = DATASETS_SH[choice]
    my_env = os.environ.copy()
    my_env.update({"DIR": download_dir})
    subprocess.run(['bash', file_path], env=my_env)

def convert(load_dir: str, save_dir: str, tokenizer_path: str, workers: int, model_id: str) -> None:
    file_path = DATA_CONVERT_SH[model_id]
    my_env = os.environ.copy()
    my_env.update({
        "INPUT_FILE": load_dir,
        "TOKENIZER_PATH": tokenizer_path,
        "OUTPUT_PREFIX": save_dir,
        "WORKERS": str(workers),
    })
    subprocess.run(['bash', file_path], env=my_env)