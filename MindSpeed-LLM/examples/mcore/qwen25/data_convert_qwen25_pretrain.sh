source /usr/local/Ascend/cann/set_env.sh
mkdir -p ${OUTPUT_PREFIX}

python ./preprocess_data.py \
	--input ${INPUT_FILE} \
	--tokenizer-name-or-path ${TOKENIZER_PATH} \
	--output-prefix ${OUTPUT_PREFIX} \
	--tokenizer-type PretrainedFromHF \
	--workers ${WORKERS} \
	--log-interval 1000