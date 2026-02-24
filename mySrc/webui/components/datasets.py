from ...extras.packages import is_gradio_available
if is_gradio_available():
    import gradio as gr
from ...handler.datasets import download, convert
from ...extras.error import validate_value

def build_datasets_tab() -> None:
    with gr.Column():
        gr.Markdown("### 数据集下载")
        with gr.Row():
            name = gr.Dropdown(
                choices=["alpaca", "enwiki", "c4"],
                label="数据集选择",
                value="alpaca",
                interactive=True
            )
        download_btn = gr.Button("下载")
        download_btn.click(
            fn=download,
            inputs=[name],
            outputs=[]
        )
        gr.Markdown("### 数据集格式转换")
        with gr.Row():
            load_dir = gr.Textbox(
                label="原始路径",
                placeholder="请输入目标文件或文件夹路径",
                value="D:/datasets/src/alpaca/",
                interactive=True
            )
            save_dir = gr.Textbox(
                label="保存路径",
                placeholder="请输入目标文件夹路径",
                value="D:/datasets/dst/alpaca/",
                interactive=True
            )
        def update_load_dir(name: str) -> gr.Textbox:
            return gr.update(value=f"D:/datasets/src/{name}/")
        def update_save_dir(name: str) -> gr.Textbox:
            return gr.update(value=f"D:/datasets/dst/{name}/")
        name.change(
            fn=update_load_dir,
            inputs=[name],
            outputs=[load_dir]
        )
        name.change(
            fn=update_save_dir,
            inputs=[name],
            outputs=[save_dir]
        )
        with gr.Row():
            tokenizer_path = gr.Textbox(
                label="目标模型的tokenizer原数据文件夹",
                placeholder="请输入目标模型转换前的文件夹路径",
                value="D:/model_from_hf/qwen2.5-7b-hf/",
                interactive=True
            )
            workers = gr.Number(
                label="多线程处理数",
                precision=0,
                value=4,
                interactive=True
            )
        convert_btn = gr.Button("转换")
        convert_btn.click(
            fn=convert,
            inputs=[load_dir, save_dir, tokenizer_path, workers],
            outputs=[]
        )