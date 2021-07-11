#!/usr/bin/env zsh

# AT remove label(0-9)
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-svlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --seed 1 --save-label [3,5]
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --seed 1 --save-label [2,3,4,5]
wait
echo "结束训练......"









