source /usr/local/Ascend/cann/set_env.sh
INPUT_FILE=${INPUT_FILE:-"./datasets/src/alpaca"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"../../autodl-tmp/model_from_hf/qwen2.5-0point5b-hf/"}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"./datasets/dst/alpaca"}
WORKERS=${WORKERS:-4}
python ./preprocess_data.py \
	--input ${INPUT_FILE} \
	--tokenizer-name-or-path ${TOKENIZER_PATH} \
	--output-prefix ${OUTPUT_PREFIX} \
	--tokenizer-type PretrainedFromHF \
	--workers ${WORKERS} \
	--log-interval 1000