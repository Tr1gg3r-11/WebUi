# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p ${OUTPUT_PREFIX}

python ./preprocess_data.py \
    --input ${INPUT_FILE} \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix ${OUTPUT_PREFIX} \
    --json-keys text \
    --workers ${WORKERS} \
    --log-interval 1000