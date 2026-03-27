# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf llama2 \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size ${TP} \
    --target-pipeline-parallel-size ${PP} \
    --add-qkv-bias \
    --lora-r ${LORA_R} \
    --lora-alpha ${LORA_ALPHA} \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --load-dir ${LOAD_DIR} \
    --lora-load ${LORA_LOAD} \
    --save-dir ${ORI_DIR}