# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ${LOAD_DIR} \
    --save-dir ${SAVE_DIR} \
    --hf-cfg-dir ${ORI_DIR} \
    --model-type-hf qwen3