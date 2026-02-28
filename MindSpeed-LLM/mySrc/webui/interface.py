#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from .css import CSS
from .components.weights import build_weights_tab
from .components.datasets import build_datasets_tab
from .components.train import build_train_config_tab
from .components.monitor import build_monitor_tab
from ..extras.misc import fix_proxy, is_env_enabled
from ..extras.packages import is_gradio_available
if is_gradio_available():
    import gradio as gr
from ..extras.constants import status_map
from ..handler.train import train_monitor

def create_ui() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="全流程训练控制面板") as demo:
        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown("## Mindspeed-LLm训练WebUi\n覆盖模型/数据准备、训练配置与监控")
            with gr.Column(scale=1, min_width=150):
                status_indicator = gr.HTML(
                    value=status_map['idle'],
                    visible=True
                )
        with gr.Tabs() as tabs:
            with gr.TabItem("模型权重下载与格式转换", id=0):
                build_weights_tab()
            with gr.TabItem("数据集下载与格式转换", id=1):
                build_datasets_tab()
            with gr.TabItem("训练任务配置", id=2):
                build_train_config_tab(tabs, status_indicator)
            with gr.TabItem("训练过程数据展示与格式转换", id=3):
                build_monitor_tab(status_indicator)
        timer = gr.Timer(value=2)
        timer.tick(
            fn=train_monitor,
            outputs=status_indicator,
        )
    return demo

def run_web_ui() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    # server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    server_name ="127.0.0.1"
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    fix_proxy(ipv6_enabled=gradio_ipv6)
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)
