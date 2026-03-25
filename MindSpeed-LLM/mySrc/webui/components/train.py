from ...extras.packages import is_gradio_available
if is_gradio_available():
    import gradio as gr
from ...extras.constants import SUPPORTED_MODEL, LR, MIN_LR
from ...handler.train import get_train_config
def build_train_config_tab(tabs: gr.Tabs, status_indicator: gr.HTML) -> None:
    with gr.Column():
        gr.Markdown("### 训练任务配置")
        md = gr.Markdown("> 💡 提示:训练模式和多样本pack应与**数据集格式转换**时设置保持一致")
        with gr.Row():
            mode = gr.Dropdown(
                choices=["pretrain", "SFT(全参)", "SFT(LoRA)"],
                label="训练模式",
                value="pretrain",
                interactive=True
            )
            pack = gr.Checkbox(
                label="多样本pack(多个样本拼接打包成一个长序列)",
                value=False,
                interactive=True
            )
            model_id = gr.Dropdown(
                choices=SUPPORTED_MODEL,
                label="模型选择",
                value=SUPPORTED_MODEL[0],
                interactive=True
            )
            npus = gr.Number(label="每节点npu卡数", value=8, precision=0, interactive=True)
        with gr.Row():
            master_addr = gr.Textbox(
                label="节点ip",
                info="单机使用本节点ip,多机所有节点都配置为master_ip",
                value="localhost",
                interactive=True
            )
            master_port = gr.Number(label="端口号", value=6000, precision=0, interactive=True)
            nodes = gr.Number(label="节点数量", value=1, precision=0, interactive=True)
            node_rank = gr.Dropdown(
                choices = [0],
                label = "当前节点号",
                interactive = True
            )
            def rank_update(nodes: gr.Number):
                return gr.update(choices=list(range(nodes)))
            nodes.change(
                fn=rank_update,
                inputs=nodes,
                outputs=node_rank
            )
        with gr.Row():
            load_dir = gr.Textbox(
                label="模型转换后权重路径",
                value="./models_mcore_weights/your_model/",
                interactive=True
            )
            save_dir = gr.Textbox(
                label="模型训练后权重保存路径",
                value="./ckpt/your_model",
                interactive=True
            )
        with gr.Row():
            data_path = gr.Textbox(
                label="数据集转换后路径",
                value="./datasets/dst/your_dataset/your_dataset_text_document",
                interactive=True
            )
            tokenizer_path = gr.Textbox(
                label="模型原始tokenizer路径",
                value="./models_from_hf/your_model/",
                interactive=True
            )
        md = gr.Markdown("> 💡 提示:此处并行设置应与**模型权重转换**时设置保持一致")
        with gr.Row():
            tp = gr.Number(label="tensor-parallel-size", value=1, precision=0, interactive=True)
            pp = gr.Number(label="pipeline-parallel-size", value=4, precision=0, interactive=True)
            cp = gr.Number(label="context-parallel-size", value=1, precision=0, interactive=True)
            seq_len = gr.Number(label="seq_length(若数据转换时设置过, 应不小于设置值)", value=4096, precision=0, interactive=True)
            mbs = gr.Number(label="micro-batch-size", info="微批次大小,决定每次前向/反向传播处理的样本数量。较大的值可以提高GPU利用率,但会增加内存使用", value=1, precision=0, interactive=True)
            gbs = gr.Number(label="global-batch-size", info="全局批次大小,是一次迭代更新的总样本数", value=64, precision=0, interactive=True)
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
        with gr.Row(visible=False) as row1:
            with gr.Column():
                md1 = gr.Markdown("> 表示低秩矩阵的维度。较低的 rank 值模型在训练时会使用更少的参数更新")
                lora_r = gr.Number(label="lora_r", value=8, precision=0, interactive=True)
            with gr.Column():
                md2 = gr.Markdown("> 控制 LoRA 权重对原始权重的影响比例, 数值越高则影响越大。一般保持 alpha/r 为 2")
                lora_alpha = gr.Number(label="lora_alpha", value=16, precision=0, interactive=True)
            with gr.Column():
                md3 = gr.Markdown("> 是否启用CCLoRA算法，该算法通过计算通信掩盖提高性能")
                lora_fusion = gr.Checkbox(label="lora_fusion", interactive=True, value=True)
        def row_visible(mode):
            return gr.update(visible=(mode=="SFT(LoRA)"))
        mode.change(fn=row_visible,inputs=mode,outputs=row1)
        with gr.Row():
            train_iters = gr.Number(label="train_iters", value=2000, precision=0, interactive=True)
            with gr.Column():
                lr = gr.Number(label="学习率", value=1.25e-6, interactive=True)
        def lr_update(model_id: gr.Dropdown, mode : gr.Dropdown, pack: gr.Checkbox):
            target = f"{model_id}_{mode}"
            if pack and mode == "SFT(全参)":
                target += "_pack"
            if target in MIN_LR:
                return gr.update(value=LR[target], label=f"学习率(不得小于{MIN_LR[target]})")
            return gr.update(value=LR[target])
        gr.on(
            triggers=[mode.change, model_id.change, pack.change],
            fn=lr_update,
            inputs=[model_id, mode, pack],
            outputs=lr
        )
        train_btn = gr.Button("开始训练")
        train_btn.click(
            fn=get_train_config,
            inputs=[pack, model_id, mode, npus, master_addr, master_port, nodes, node_rank, load_dir, save_dir, data_path, tokenizer_path, tp, pp , cp, seq_len, mbs, gbs, train_iters, lr, lora_r, lora_alpha, lora_fusion],
            outputs=[tabs, status_indicator]
        )