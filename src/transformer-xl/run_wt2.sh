#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../../data/wikitext-2/ \
        --dataset wt2 \
        --n_layer 16 \
        --d_model 512 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --log-interval 100 \
        --eval-interval 1000 \
        --max_step 2000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 100 \
        --batch_size 32 \
        --gpu0_bsz 4 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../../data/wikitext-2/ \
        --dataset wt2 \
        --tgt_len 64 \
        --mem_len 640 \
        --clamp_len 400 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
