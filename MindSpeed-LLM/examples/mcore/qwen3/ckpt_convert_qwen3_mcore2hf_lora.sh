# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size ${TP} \
    --target-pipeline-parallel-size ${PP} \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --lora-r ${LORA_R} \
    --lora-alpha ${LORA_ALPHA} \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --load-dir ${LOAD_DIR} \
    --lora-load ${LORA_LOAD} \
    --save-dir ${ORI_DIR} \
    --model-type-hf qwen3
