# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ${INPUT_FILE} \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --output-prefix ${OUTPUT_PREFIX} \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers ${WORKERS} \
    --log-interval 1000 \
    --prompt-type qwen
