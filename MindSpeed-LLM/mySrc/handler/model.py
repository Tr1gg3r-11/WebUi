from ..extras.constants import PP_SUPPORTED_MODEL, TP_SUPPORTED_MODEL, CP_SUPPORTED_MODEL, MODEL_DOWNLOAD_SH, MODEL_CONVERT_HF2MCORE_SH, MODEL_CONVERT_MCORE2HF_SH, MODEL_CONVERT_MCORE2HF_LORA_SH
from ..extras.error import validate_value
import subprocess
import os
def download(platform: str, model_id: str, cache_dir: str) -> None:
    target = model_id+'-'+platform
    file_path = MODEL_DOWNLOAD_SH[target]
    my_env = os.environ.copy()
    my_env.update({"DIR": cache_dir})
    subprocess.run(['bash', file_path], env=my_env)

def convert_hf2mcore(load_dir: str, save_dir: str, model_id: str, tp: int, cp: int, pp: int) -> None:
    values = [tp, cp, pp]
    keys = ['TP', 'CP', 'PP']
    check = True
    for k,v in zip(keys, values):
        check = check and validate_value(v, k)
        if not check:
            return
    file_path = MODEL_CONVERT_HF2MCORE_SH[model_id]
    my_env = os.environ.copy()
    my_env.update({"TP": str(tp), "PP": str(pp), "CP": str(cp), "LOAD_DIR": load_dir, "SAVE_DIR": save_dir})
    subprocess.run(['bash', file_path], env=my_env)

def convert_mcore2hf(load_dir: str, save_dir: str, model_id: str, tp: int, cp: int, pp: int) -> None:
    values = [tp, cp, pp]
    keys = ['TP', 'CP', 'PP']
    check = True
    for k,v in zip(keys, values):
        check = check and validate_value(v, k)
        if not check:
            return
    file_path = MODEL_CONVERT_MCORE2HF_SH[model_id]
    my_env = os.environ.copy()
    my_env.update({"TP": str(tp), "PP": str(pp), "CP": str(cp), "LOAD_DIR": load_dir, "SAVE_DIR": save_dir})
    subprocess.run(['bash', file_path], env=my_env)