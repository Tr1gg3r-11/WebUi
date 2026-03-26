#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=${NPUS_PER_NODE}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
NNODES=${NNODES}
NODE_RANK=${NODE_RANK}
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_SAVE_DIR=${SAVE_DIR}
DATA_PATH=${DATA_PATH}
TOKENIZER_PATH=${TOKENIZER_PATH}
CKPT_LOAD_DIR=${LOAD_DIR}

TP=${TP}
PP=${PP}
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
CP_ALGO=megatron_cp_algo

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

LORA_FUSION=${LORA_FUSION:-false}
TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen \
    --no-pad-to-seq-lengths \
    --padded-samples \
    --lora-r 8 \
    --lora-alpha 16 \
    $( [ "$LORA_FUSION" = "true" ] && echo "--lora-fusion" ) \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-distributed-optimizer \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --group-query-attention \
    --num-query-groups 2 \
    --num-layers 24 \
    --hidden-size 896 \
    --ffn-hidden-size 4864 \
    --num-attention-heads 14 \
    --rotary-base 1000000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --swiglu \
    --add-qkv-bias \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --lr ${LR} \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --init-method-std 0.01 \
    --position-embedding-type rope \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 7.75e-8 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    --distributed-backend nccl \
    --log-throughput \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --transformer-impl local \
    | tee logs/tune_mcore_qwen25_0point5b_4k_lora.log
