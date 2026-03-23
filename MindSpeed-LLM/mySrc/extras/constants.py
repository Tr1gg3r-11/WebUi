RUNNING_LOG = "running_log.txt"

SUPPORTED_MODEL=["Qwen2.5-0.5B", "Qwen2.5-7B", "Test_Other"]
# TP_SUPPORTED_MODEL = {"Qwen2.5-0.5B", "Qwen2.5-7B"}
# PP_SUPPORTED_MODEL = {"Qwen2.5-0.5B", "Qwen2.5-7B"}
# CP_SUPPORTED_MODEL = {"Qwen2.5-0.5B"}

SFT_FULL = {}
SFT_LORA = {}
PRETRAIN = {}

MINIMUM = {
    'TP' : 1,
    'PP' : 1,
    'CP' : 1,
    '每节点npu卡数' : 1,
    '多线程处理数' : 1,
    '端口号' : 0,
    '节点数量' : 1,
    'seq_length' : 1,
    'micro-batch-size': 1,
    'global-batch-size': 1,
    'train_iters' : 1,
    '学习率' : 0,
    'lora_r' : 1,
    'lora_alpha' : 1
}
MAXIMUM = {
}
LR = {
    'Qwen2.5-0.5B_pretrain' : 1.25e-6,
    'Qwen2.5-0.5B_SFT(全参)' : 1e-6,
    'Qwen2.5-0.5B_SFT(LoRA)' : 7.75e-7,
    'Qwen2.5-0.5B_SFT(全参)_pack' : 7.75e-7
}
MIN_LR = {
    'Qwen2.5-0.5B_pretrain' : 1.25e-7,
    'Qwen2.5-0.5B_SFT(LoRA)' : 7.75e-8,
    'Qwen2.5-0.5B_SFT(全参)_pack' : 7.75e-8
}

status_map = {
    "idle": '<div style="padding: 10px; border-radius: 5px; background-color: #f0f0f0; color: #666;">⚪ 未训练</div>',
    "stopping": '<div style="padding: 10px; border-radius: 5px; background-color: #fff3cd; color: #856404;">🟡 训练停止中</div>',
    "training": '<div style="padding: 10px; border-radius: 5px; background-color: #d4edda; color: #155724;">🟢 训练中</div>',
    "completed": '<div style="padding: 10px; border-radius: 5px; background-color: #d1ecf1; color: #0c5460;">🔵 训练完成</div>',
    "stopped": '<div style="padding: 10px; border-radius: 5px; background-color: #f8d7da; color: #721c24;">🔴 训练已停止</div>'
}

DATASET_SEQ=[
    "Qwen2.5-0.5B_SFT(全参)_pack"
]

DATASETS_SH={
    "alpaca-HuggingFace":"mySrc/scripts/datasets/alpaca-HuggingFace.sh",
    "alpaca-ModelScope":"mySrc/scripts/datasets/alpaca-ModelScope.sh",
}
DATA_CONVERT_SH={
    "Qwen2.5-0.5B_pretrain":"examples/mcore/qwen25/data_convert_qwen25_pretrain.sh",
    "Qwen2.5-0.5B_sft":"examples/mcore/qwen25/data_convert_qwen25_instruction.sh",
    "Qwen2.5-0.5B_sft_pack":"examples/mcore/qwen25/data_convert_qwen25_instruction_pack.sh",
}
MODEL_DOWNLOAD_SH={
    "Qwen2.5-0.5B-HuggingFace":"mySrc/scripts/models/qwen2.5-0point5b-HuggingFace.sh",
    "Qwen2.5-0.5B-ModelScope":"mySrc/scripts/models/qwen2.5-0point5b-ModelScope.sh",
}
MODEL_CONVERT_HF2MCORE_SH={
    "Qwen2.5-0.5B":"examples/mcore/qwen25/ckpt_convert_qwen25_hf2mcore.sh",
}
MODEL_CONVERT_MCORE2HF_SH={
    "Qwen2.5-0.5B":"examples/mcore/qwen25/ckpt_convert_qwen25_mcore2hf.sh",
}
MODEL_CONVERT_MCORE2HF_LORA_SH={
    "Qwen2.5-0.5B":"examples/mcore/qwen25/ckpt_convert_qwen25_mcore2hf_lora.sh",
}
PRETRAIN_SH={
    "Qwen2.5-0.5B":"examples/mcore/qwen25/pretrain_qwen25_0point5_32k_ptd.sh",
    "Qwen2.5-0.5B_pack":"examples/mcore/qwen25/pretrain_qwen25_0point5_32k_ptd.sh",
}
SFT_LORA_SH={
    "Qwen2.5-0.5B":"examples/mcore/qwen25/tune_qwen25_0point5b_4k_lora_ptd.sh",
    "Qwen2.5-0.5B_pack":"examples/mcore/qwen25/tune_qwen25_0point5b_4k_lora_ptd.sh",
}
SFT_SH={
    "Qwen2.5-0.5B":"examples/mcore/qwen25/tune_qwen25_0point5b_4k_full_ptd.sh",
    "Qwen2.5-0.5B_pack":"examples/mcore/qwen25/tune_qwen25_0point5b_4k_full_pack.sh",
}