source /usr/local/Ascend/cann/set_env.sh  # 修改为实际安装的Toolkit包路径

python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 2 \
       --add-qkv-bias \
       --load-dir ../autodl-tmp/model_from_hf/qwen2.5-0point5b-hf/ \
       --save-dir ../autodl-tmp/model_weights/qwen2.5_mcore/ \
       --tokenizer-model ./model_from_hf/qwen2.5-0point5b-hf/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16