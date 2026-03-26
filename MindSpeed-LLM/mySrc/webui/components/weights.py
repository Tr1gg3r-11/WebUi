from ...extras.packages import is_gradio_available
if is_gradio_available():
    import gradio as gr
from ...extras.constants import SUPPORTED_MODEL
from ...handler.model import download, convert_hf2mcore

def build_weights_tab() -> None:
    with gr.Column():
        gr.Markdown("### 模型权重下载与格式转换")
        gr.Markdown("#### 权重下载")
        with gr.Row():
            platform = gr.Dropdown(
                choices=["HuggingFace", "ModelScope"],
                label="模型来源",
                value="HuggingFace",
                interactive=True
            )
            model_id = gr.Dropdown(
                choices=SUPPORTED_MODEL,
                label="模型选择",
                value=SUPPORTED_MODEL[0],
                interactive=True
            )
        with gr.Row():
            cache_dir = gr.Textbox(
                label="下载目录",
                placeholder="请输入文件夹路径",
                value=f"./models_from_hf/your_model/",
                interactive=True
            )
        def update_download_dir(model_id: str) -> gr.Textbox:
            return gr.update(value=f"./models_from_hf/{model_id}/")
        model_id.change(
            fn=update_download_dir,
            inputs=[model_id],
            outputs=[cache_dir]
        )
        download_btn = gr.Button("下载")
        download_btn.click(
            fn=download,
            inputs=[platform, model_id, cache_dir],
            outputs=[]
        )
        gr.Markdown("#### 权重转换")
        md = gr.Markdown("> 💡 提示:TPxPPxCP 不应超过**可用GPU总数**(不可配置时默认为1)")
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
        with gr.Row():
            load_dir = gr.Textbox(
                label="加载目录",
                placeholder="请输入文件夹路径",
                value=f"./models_from_hf/your_model/",
                interactive=True
            )
            save_dir = gr.Textbox(
                label="保存目录",
                placeholder="请输入文件夹路径",
                value=f"./models_mcore_weights/your_model/",
                interactive=True
            )
        model_id.change(
            fn=update_download_dir,
            inputs=[model_id],
            outputs=[load_dir]
        )
        def update_ckpt_dir(model_id: str) -> gr.Textbox:
            return gr.update(value=f"./models_mcore_weights/{model_id}/")
        model_id.change(
            fn=update_ckpt_dir,
            inputs=[model_id],
            outputs=[save_dir]
        )
        convert_btn = gr.Button("转换")
        
        convert_btn.click(
            fn=convert_hf2mcore,
            inputs=[load_dir, save_dir, model_id, tp, cp, pp]
        )