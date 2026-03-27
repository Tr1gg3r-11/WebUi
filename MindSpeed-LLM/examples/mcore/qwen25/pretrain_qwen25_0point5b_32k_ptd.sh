#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=${NPUS_PER_NODE}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
NNODES=${NNODES}
NODE_RANK=${NODE_RANK}
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR=${LOAD_DIR}
CKPT_SAVE_DIR=${SAVE_DIR}
DATA_PATH=${DATA_PATH}
TOKENIZER_PATH=${TOKENIZER_PATH}

TP=${TP}
PP=${PP}
CP=${CP}
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

GPT_ARGS="
    --use-mcore-models \
    --use-cp-send-recv-overlap \
    --use-fused-ring-attention-update \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_ALGO} \
    --sequence-parallel \
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
    --min-lr 1.25e-7 \
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

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --log-throughput \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --transformer-impl local \
    | tee logs/train_mcore_qwen25_0point5b_32k.log
