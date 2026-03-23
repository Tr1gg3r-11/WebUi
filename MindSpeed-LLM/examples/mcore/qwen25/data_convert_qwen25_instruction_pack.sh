# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p ${OUTPUT_PREFIX}

python ./preprocess_data.py \
	--input ${INPUT_FILE} \
	--tokenizer-name-or-path ${TOKENIZER_PATH} \
	--output-prefix ${OUTPUT_PREFIX} \
	--handler-name AlpacaStyleInstructionHandler \
	--tokenizer-type PretrainedFromHF \
	--workers ${WORKERS} \
	--log-interval 1000 \
	--prompt-type qwen \
	--pack \
	--neat-pack \
	--seq-length ${SEQ_LENGTH} \
#  demo提供的是单轮数据集，若使用多轮数据需要修改以下参数：
#  --input ./dataset/多轮数据集
#  --map-keys '{"prompt":"instruction","query":"input","response":"output", "history":"history"}'