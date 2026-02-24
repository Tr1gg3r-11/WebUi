from ..extras.constants import TP_SUPPORTED_MODEL,CP_SUPPORTED_MODEL,PP_SUPPORTED_MODEL
from ..extras.packages import is_gradio_available
if is_gradio_available():
    import gradio as gr
import time
import threading
from typing import Optional
from ..extras.error import validate_value
from ..extras.constants import status_map
from typing import Any

train_thread :Optional[threading.Thread] = None
stop_training = threading.Event()
training_completed = threading.Event()
training_stopped = threading.Event()
trainer_log = []
def get_train_config(model_id: str,
                     mode: str,
                     npus: int,
                     master_addr: str,
                     master_port: int,
                     nodes: int,
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
                     epochs: int,
                     lr:float) -> list[gr.Tabs, gr.HTML]:
    config = {}
    config['model_id'] = model_id
    config['train_mode'] = mode
    config['npus_pre_node'] = npus
    config['addr'] = master_addr
    config['port'] = master_port
    config['Nnodes'] = nodes
    config['load_dir'] = load_dir
    config['save_dir'] = save_dir
    config['data_path'] = data_path
    config['tokenizer_path'] = tokenizer_path
    if model_id in TP_SUPPORTED_MODEL:
        config['tp'] = tp
    if model_id in PP_SUPPORTED_MODEL:
        config['pp'] = pp
    if model_id in CP_SUPPORTED_MODEL:
        config['cp'] = cp
    config['seq_len'] = seq_len
    config['mbs'] = mbs
    config['gbs'] = gbs
    config['epochs'] = epochs
    config['lr'] = lr
    # for k,v in config.items():
    #     print(k,v)

    #check value
    keys = ['TP', 'PP', 'CP', '每节点npu卡数', '端口号', '节点数量', 'seq_length', 'micro-batch-size', 'global-batch-size', 'epochs', '学习率']
    values = [tp, pp ,cp ,npus, master_port, nodes, seq_len, mbs, gbs, epochs, lr]
    check = True
    for k,v in zip(keys,values):
        check = check and validate_value(v,k)
        if not check:
            return [gr.Tabs(selected=2), gr.skip()]


    first_update_eve = threading.Event()
    global train_thread
    if stop_training.is_set():
        gr.Info("🚀 训练任务中止中，请等待...", duration=0)
        return [gr.Tabs(selected=2), gr.skip()]
    elif train_thread and train_thread.is_alive():
        gr.Info("🚀 训练任务执行中，请先中止当前任务", duration=0)
        return [gr.Tabs(selected=2), gr.skip()]
    gr.Info("🚀 正在启动训练任务，即将跳转到监控页面...", duration=1)
    train_thread = threading.Thread(target=train, args=(config, first_update_eve))
    train_thread.daemon = True
    train_thread.start()
    first_update_eve.wait(timeout=1)

    return [gr.Tabs(selected=3), gr.update(value=status_map['training'])]
def train(config: dict[str: Any], first_update_eve: threading.Event) -> None:
    global stop_training, training_completed, training_stopped, trainer_log

    first_update_done = False
    update_steps = int(config['gbs'] / config['mbs'] / config['tp'])
    epochs = int(config['epochs'])
    global_step = 0
    trainer_log.clear()
    #training
    for epoch in range(epochs):
        for step in range(120):
            time.sleep(0.1)
            log = {}
            #loss
            loss = global_step * global_step / 10 + 2
            log['current_steps'] = global_step
            log['loss'] = loss
            trainer_log.append(log)
            if stop_training.is_set():
                stop_training.clear()
                training_stopped.set()
                return
            if step % update_steps == 0 and step > 0:
                print(f"step{step}-update")
                #update
                if not first_update_done:
                    first_update_done = True
                    first_update_eve.set()
            global_step += 1
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