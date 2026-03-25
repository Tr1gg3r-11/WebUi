from ..extras.packages import is_gradio_available
if is_gradio_available():
    import gradio as gr
import time
import threading
from typing import Optional
from ..extras.error import validate_value
from ..extras.constants import status_map, PRETRAIN_SH, SFT_LORA_SH, SFT_SH
from typing import Any
import subprocess
import os
import signal
from pathlib import Path
from datetime import datetime
LOG_FILE = Path("./training_logs/trainer_log.jsonl")

train_thread :Optional[threading.Thread] = None
stop_training = threading.Event()
training_completed = threading.Event()
training_stopped = threading.Event()
def get_train_config(shared_pack: bool,
                     model_id: str,
                     mode: str,
                     npus: int,
                     master_addr: str,
                     master_port: int,
                     nodes: int,
                     node_rank: int,
                     load_dir: str,
                     save_dir: str,
                     data_path: str,
                     tokenizer_path: str,
                     tp: int,
                     pp: int,
                     cp: int,
                     seq_len: int,
                     mbs: int,
                     gbs: int,
                     train_iters: int,
                     lr: float,
                     lora_r: int,
                     lora_alpha: int,
                     lora_fusion: bool) -> list[gr.Tabs, gr.HTML]:
    config = {}
    config['pack'] = shared_pack
    config['model_id'] = model_id
    config['mode'] = mode
    config['NPUS_PER_NODE'] = npus
    config['MASTER_ADDR'] = master_addr
    config['MASTER_PORT'] = master_port
    config['NNODES'] = nodes
    config['NODE_RANK'] = node_rank
    config['LOAD_DIR'] = load_dir
    config['SAVE_DIR'] = save_dir
    config['DATA_PATH'] = data_path
    config['TOKENIZER_PATH'] = tokenizer_path
    # if model_id in TP_SUPPORTED_MODEL:
    config['TP'] = tp
    # if model_id in PP_SUPPORTED_MODEL:
    config['PP'] = pp
    # if model_id in CP_SUPPORTED_MODEL:
    config['CP'] = cp
    config['SEQ_LEN'] = seq_len
    config['MBS'] = mbs
    config['GBS'] = gbs
    os.environ['MBS'] = str(mbs)
    os.environ['GBS'] = str(gbs)
    config['TRAIN_ITERS'] = train_iters
    config['LR'] = lr
    # for k,v in config.items():
    #     print(k,v)
    config['LORA_R'] = lora_r
    config['LORA_ALPHA'] = lora_alpha
    config['LORA_FUSION'] = lora_fusion

    #check value
    keys = ['TP', 'PP', 'CP', '每节点npu卡数', '端口号', '节点数量', 'seq_length', 'micro-batch-size', 'global-batch-size', 'train_iters', '学习率', 'lora_r', 'lora_alpha']
    values = [tp, pp ,cp ,npus, master_port, nodes, seq_len, mbs, gbs, train_iters, lr, lora_r, lora_alpha]
    check = True
    for k,v in zip(keys,values):
        check = check and validate_value(v,k)
        if not check:
            return [gr.Tabs(selected=2), gr.skip()]


    global train_thread
    if stop_training.is_set():
        gr.Info("🚀 训练任务中止中，请等待...", duration=0)
        return [gr.Tabs(selected=2), gr.skip()]
    elif train_thread and train_thread.is_alive():
        gr.Info("🚀 训练任务执行中，请先中止当前任务", duration=0)
        return [gr.Tabs(selected=2), gr.skip()]
    gr.Info("🚀 正在启动训练任务，即将跳转到监控页面...", duration=1)
    train_thread = threading.Thread(target=train, args=(config,))
    train_thread.daemon = True
    train_thread.start()

    return [gr.Tabs(selected=3), gr.update(value=status_map['training'])]
def train(config: dict[str: Any]) -> None:
    global stop_training, training_completed, training_stopped, trainer_log
    my_env = os.environ.copy()
    choice = config['model_id']
    if config['pack']:
        choice += "_pack"
    if config['mode'] == "pretrain":
        file_path = PRETRAIN_SH[choice]
        for k,v in config.items():
            if not k in ['mode', 'model_id', 'pack']:
                my_env[k] = str(v)
        process = subprocess.Popen(['bash', file_path], env=my_env, preexec_fn=os.setsid)
        while process.poll() is None:
            time.sleep(0.1)
    elif config['mode'] == "SFT(LoRA)":
        file_path = SFT_LORA_SH[choice]
        for k,v in config.items():
            if not k in ['mode', 'model_id', 'pack']:
                my_env[k] = str(v)
        process = subprocess.Popen(['bash', file_path], env=my_env, preexec_fn=os.setsid)
        while process.poll() is None:
            time.sleep(0.1)
    else:
        file_path = SFT_SH[choice]
        for k,v in config.items():
            if not k in ['mode', 'model_id', 'pack']:
                my_env[k] = str(v)
        process = subprocess.Popen(['bash', file_path], env=my_env, preexec_fn=os.setsid)
        while process.poll() is None:
            time.sleep(0.1)
    if LOG_FILE.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = LOG_FILE.parent / f"{config['model_id']}_{config['mode']}_{timestamp}.jsonl"
        os.rename(LOG_FILE, new_name)
    training_completed.set()
def stop() -> gr.HTML:
    global stop_training
    if train_thread.is_alive():
        stop_training.set()
        return gr.update(value=status_map['stopping'])
    gr.Warning(f"⚠ 当前没有训练任务!")
    return gr.skip()
def train_monitor() -> gr.HTML:
    if training_completed.is_set():
        training_completed.clear()
        return gr.update(value=status_map['completed'])
    elif training_stopped.is_set():
        training_stopped.clear()
        return gr.update(value=status_map['stopped'])
    else :
        return gr.skip()