#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../../data/enwik8/ \
        --dataset enwik8 \
        --n_layer 12 \
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
        --eval-interval 999 \
        --max_step 1000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 64 \
        --batch_size 22 \
        --gpu0_bsz 4 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../../data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 80 \
        --mem_len 150 \
        --clamp_len 150 \
        --same_length \
        --split test \
        --work_dir ./LM-TFM-enwik8/20211002-163828/
        ${@:2}
else
    echo 'unknown argment 1'
fi
