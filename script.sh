#!/bin/bash

rm -rf ./results

for seed in 0 1 2 3 4; do
  python train.py \
    --run_name naive \
    --seed $seed \
    --lr 0.0001 \
    --epsilon_end 0.01 \
    --epsilon_decay 200000 \
    --batch_size 32 \
    --warmup_steps 10000 \
    --total_steps 1000000

    python train.py \
    --use_target_network \
    --run_name tn \
    --seed $seed \
    --lr 0.0001 \
    --epsilon_end 0.01 \
    --epsilon_decay 200000 \
    --target_update_freq 500 \
    --batch_size 32 \
    --warmup_steps 10000 \
    --total_steps 1000000

     python train.py \
    --use_replay_buffer \
    --run_name er \
    --seed $seed \
    --lr 0.0001 \
    --epsilon_end 0.01 \
    --epsilon_decay 200000 \
    --update_every 4 \
    --batch_size 32 \
    --buffer_size 100000 \
    --warmup_steps 10000 \
    --total_steps 1000000

  python train.py \
    --use_target_network \
    --use_replay_buffer \
    --run_name tn_er \
    --seed $seed \
    --lr 0.0001 \
    --epsilon_end 0.01 \
    --epsilon_decay 200000 \
    --update_every 4 \
    --target_update_freq 5000 \
    --batch_size 32 \
    --buffer_size 100000 \
    --warmup_steps 10000 \
    --total_steps 1000000
done