from ...extras.packages import is_gradio_available
if is_gradio_available():
    import gradio as gr
from ...handler.train import stop
from ...handler.model import convert_mcore2hf
from ...extras.constants import SUPPORTED_MODEL
from ...extras.ploting import gen_loss_plot
import time
import pickle
from pathlib import Path
import os
LOG_FILE = Path("./training_logs/trainer_log.jsonl")
trainer_log = []
def load_log_from_file():
    if not LOG_FILE.exists():
        print("日志文件不存在")
        return
    
    with open(LOG_FILE, 'rb') as f:
        try:
            global trainer_log
            trainer_log = pickle.load(f)
            merge_log()
        except Exception as e:
            print(f"读取失败: {e}")
def merge_log():
    global trainer_log
    MBS = int(os.environ['MBS'])
    GBS = int(os.environ['GBS'])
    threshold = int(GBS / MBS)
    # print('threshold: ', threshold)
    ct = 0
    loss = []
    step = 1
    new_trainer_log = []
    for log in trainer_log:
        loss.append(log['loss'])
        ct += 1
        if ct == threshold:
            ct = 0
            new_log = {}
            new_log['loss'] = sum(loss) / len(loss)
            new_log['current_steps'] = step
            step += 1
            loss = []
            ct = 0
            new_trainer_log.append(new_log)
    trainer_log = new_trainer_log

            
def build_monitor_tab() -> None:
    with gr.Column():
        gr.Markdown("### 训练过程数据展示")
        with gr.Row():
            with gr.Column():
                gap = gr.Number(label="刷新间隔 (秒)", value=5, precision=0, interactive=True)
                auto_refresh_cb = gr.Checkbox(label="自动刷新", value=False, interactive=True)
            with gr.Column():
                manual_refresh = gr.Button("手动刷新")
        with gr.Row():
            loss_plot = gr.Plot(label="损失曲线")
        def refresh() -> gr.Plot:
            load_log_from_file()
            return gen_loss_plot(trainer_log)
        manual_refresh.click(
            fn=refresh,
            outputs=loss_plot
        )
        def auto_refresh(auto_refresh_cb: gr.Checkbox, gap: gr.Number, last_refresh: gr.State) -> list[gr.Plot, gr.State]:
            if not auto_refresh_cb:
                return [gr.skip(), last_refresh]
            current_time = time.time()
            if last_refresh == 0:
                return [gr.skip(), current_time]
            if current_time - last_refresh < gap:
                return [gr.skip(), last_refresh]
            load_log_from_file()
            return [gen_loss_plot(trainer_log), current_time]
        last_refresh = gr.State(value=0.0)
        timer = gr.Timer(value=1)
        timer.tick(
            fn=auto_refresh,
            inputs=[auto_refresh_cb, gap, last_refresh],
            outputs=[loss_plot, last_refresh]
        )
        def reset_time(auto_refresh_cb: gr.Checkbox) -> gr.State:
            if not auto_refresh_cb:
                return 0.0
            return gr.skip()
        auto_refresh_cb.change(
            fn=reset_time,
            inputs=auto_refresh_cb,
            outputs=last_refresh
        )
        

        # stop_btn = gr.Button("终止训练")
        # stop_btn.click(fn=stop, outputs=status_indicator)
        gr.Markdown("### 格式转换")
        with gr.Row():
            model_id = gr.Dropdown(
                choices=SUPPORTED_MODEL,
                label="模型选择",
                value=SUPPORTED_MODEL[0],
                interactive=True
            )
            load_dir = gr.Textbox(
                label="加载目录",
                placeholder="请输入文件夹路径",
                value=f"./models_mcore_weights/your_model/",
                interactive=True
            )
            ori_dir = gr.Textbox(
                label="模型原始config目录",
                placeholder="请输入文件夹路径",
                value=f"./models_from_hf/your_model/",
                interactive=True
            )
            save_dir = gr.Textbox(
                label="模型保存目录",
                placeholder="请输入文件夹路径",
                value=f"./models_trained/your_model/",
                interactive=True
            )
        def update_load_dir(model_id: str) -> gr.Textbox:
            return gr.update(value=f"./models_mcore_weights/{model_id}/")
        def update_save_dir(model_id: str) -> gr.Textbox:
            return gr.update(value=f"./models_trained/{model_id}/")
        def update_ori_dir(model_id: str) -> gr.Textbox:
            return gr.update(value=f"./models_from_hf/{model_id}/")
        model_id.change(
            fn=update_load_dir,
            inputs=model_id,
            outputs=load_dir
        )
        model_id.change(
            fn=update_save_dir,
            inputs=model_id,
            outputs=save_dir
        )
        model_id.change(
            fn=update_ori_dir,
            inputs=model_id,
            outputs=ori_dir
        )
        with gr.Row():
            lora = gr.Checkbox(label="lora", value=False, interactive=True)
        with gr.Row(visible=False) as row_lora:
            with gr.Column():
                md1 = gr.Markdown("> 表示低秩矩阵的维度。较低的 rank 值模型在训练时会使用更少的参数更新")
                lora_r = gr.Number(label="lora_r", value=8, precision=0, interactive=True)
            with gr.Column():
                md2 = gr.Markdown("> 控制 LoRA 权重对原始权重的影响比例, 数值越高则影响越大。一般保持 alpha/r 为 2")
                lora_alpha = gr.Number(label="lora_alpha", value=16, precision=0, interactive=True)
            with gr.Column():
                md = gr.Markdown("> !!!提示:此处lora配置应与训练时配置相同")
                lora_load=gr.Textbox(label="lora训练后的checkpoint目录", value=f"./tune_lora_ckpt/your_model/", interactive=True)
        def lora_config_update(lora: bool) -> gr.Row:
            return gr.update(visible=lora)
        lora.change(fn=lora_config_update,inputs=lora,outputs=row_lora)
        def save_visible(lora: bool) -> gr.Textbox:
            return gr.udpate(visible=not lora)
        lora.change(fn=save_visible,inputs=lora,outputs=save_dir)
        def lora_load_update(model_id: str) -> gr.Textbox:
            return gr.update(label="lora训练后的checkpoint目录", value=f"./tune_lora_ckpt/{model_id}/")
        
        md = gr.Markdown("> 💡 提示:此处应与最初下载时配置相同")
        with gr.Row():
            tp = gr.Number(
                label="TP",
                info="tensor-parallel-size",
                value=1,
                precision=0,
                interactive=True
            )
            pp = gr.Number(
                label="PP",
                info="pipeline-parallel-size",
                value=4,
                precision=0,
                interactive=True
            )
            cp = gr.Number(
                label="CP",
                info="context-parallel-size",
                value=1,
                precision=0,
                interactive=True
            )
        # def update_parallel_visibility(model_id: str) -> list[gr.Number, gr.Number, gr.Number, gr.Markdown]:
        #     tp_visible = model_id in TP_SUPPORTED_MODEL
        #     pp_visible = model_id in PP_SUPPORTED_MODEL
        #     cp_visible = model_id in CP_SUPPORTED_MODEL
        #     return [
        #         gr.update(visible=tp_visible),
        #         gr.update(visible=pp_visible),
        #         gr.update(visible=cp_visible),
        #         gr.update(visible=tp_visible or pp_visible or cp_visible)
        #     ]
        
        # model_id.change(
        #     fn=update_parallel_visibility,
        #     inputs=[model_id],
        #     outputs=[tp, pp, cp, md]
        # )
        convert_btn = gr.Button("开始转换")
        convert_btn.click(
            fn=convert_mcore2hf,
            inputs=[lora_load, lora, ori_dir, load_dir, save_dir, model_id, tp, cp, pp, lora_r, lora_alpha]
        )