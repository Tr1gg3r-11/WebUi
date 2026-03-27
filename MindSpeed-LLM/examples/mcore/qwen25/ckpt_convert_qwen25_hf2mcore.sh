source /usr/local/Ascend/cann/set_env.sh  # 修改为实际安装的Toolkit包路径

python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size ${TP} \
       --target-pipeline-parallel-size ${PP} \
       --add-qkv-bias \
       --load-dir ${LOAD_DIR} \
       --save-dir ${SAVE_DIR} \
       --tokenizer-model ${LOAD_DIR}/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16