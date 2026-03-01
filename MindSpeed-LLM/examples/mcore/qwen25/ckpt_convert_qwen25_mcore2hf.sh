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
    --load-dir ${LOAD_DIR} \
    --save-dir ${SAVE_DIR}/  # 需要填入原始HF模型路径，新权重会存于 ${SAVE_DIR}/mg2hg/