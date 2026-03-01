from ..extras.constants import PP_SUPPORTED_MODEL, TP_SUPPORTED_MODEL, CP_SUPPORTED_MODEL, MODEL_DOWNLOAD_SH
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
    res = f"hf2mcore: 将{load_dir}处模型{model_id}权重转换至{save_dir}"
    if model_id in TP_SUPPORTED_MODEL:
        res += f"tp={tp}"
    if model_id in PP_SUPPORTED_MODEL:
        res += f"pp={pp}"
    if model_id in CP_SUPPORTED_MODEL:
        res += f"cp={cp}"
    print(res)

def convert_mcore2hf(load_dir: str, save_dir: str, model_id: str, tp: int, cp: int, pp: int) -> None:
    values = [tp, cp, pp]
    keys = ['TP', 'CP', 'PP']
    check = True
    for k,v in zip(keys, values):
        check = check and validate_value(v, k)
        if not check:
            return
    res = f"mcore2hf: 将{load_dir}处模型{model_id}权重转换至{save_dir}"
    if model_id in TP_SUPPORTED_MODEL:
        res += f"tp={tp}"
    if model_id in PP_SUPPORTED_MODEL:
        res += f"pp={pp}"
    if model_id in CP_SUPPORTED_MODEL:
        res += f"cp={cp}"
    print(res)