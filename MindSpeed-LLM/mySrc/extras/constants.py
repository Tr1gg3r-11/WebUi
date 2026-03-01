RUNNING_LOG = "running_log.txt"

SUPPORTED_MODEL=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B", "Test_Other"]
TP_SUPPORTED_MODEL = {"Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B"}
PP_SUPPORTED_MODEL = {"Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B"}
CP_SUPPORTED_MODEL = {"Qwen/Qwen2.5-0.5B"}

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
    'epochs' : 1,
    '学习率' : 0
}
MAXIMUM = {
}

status_map = {
    "idle": '<div style="padding: 10px; border-radius: 5px; background-color: #f0f0f0; color: #666;">⚪ 未训练</div>',
    "stopping": '<div style="padding: 10px; border-radius: 5px; background-color: #fff3cd; color: #856404;">🟡 训练停止中</div>',
    "training": '<div style="padding: 10px; border-radius: 5px; background-color: #d4edda; color: #155724;">🟢 训练中</div>',
    "completed": '<div style="padding: 10px; border-radius: 5px; background-color: #d1ecf1; color: #0c5460;">🔵 训练完成</div>',
    "stopped": '<div style="padding: 10px; border-radius: 5px; background-color: #f8d7da; color: #721c24;">🔴 训练已停止</div>'
}

DATASETS_SH={
    "alpaca-HuggingFace":"mySrc/scripts/datasets/alpaca-HuggingFace.sh",
    "alpaca-ModelScope":"mySrc/scripts/datasets/alpaca-ModelScope.sh",
}
DATA_CONVERT_SH={
    "Qwen/Qwen2.5-0.5B":"examples/mcore/qwen25/data_convert_qwen25_pretrain.sh",
}
MODEL_DOWNLOAD_SH={
    "Qwen/Qwen2.5-0.5B-HuggingFace":"mySrc/scripts/models/qwen2.5-0point5b-HuggingFace.sh",
    "Qwen/Qwen2.5-0.5B-ModelScope":"mySrc/scripts/models/qwen2.5-0point5b-ModelScope.sh",
}
MODEL_CONVERT_HF2MCORE_SH={
    "Qwen/Qwen2.5-0.5B":"examples/mcore/qwen25/ckpt_convert_qwen25_hf2mcore.sh",
}
MODEL_CONVERT_MCORE2HF_SH={
    "Qwen/Qwen2.5-0.5B":"examples/mcore/qwen25/ckpt_convert_qwen25_mcore2hf.sh",
}
MODEL_CONVERT_MCORE2HF_LORA_SH={
    "Qwen/Qwen2.5-0.5B":"examples/mcore/qwen25/ckpt_convert_qwen25_mcore2hf_lora.sh",
}