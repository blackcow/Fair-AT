#!/usr/bin/env zsh

# 使用 3，5；2-5 来做 AT 的 FC 层 finetuning
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-ft2.py --model preactresnet --AT-method TRADES --batch-size 128 --finetune --save-label 3 5
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-ft2.py --model preactresnet --AT-method TRADES --batch-size 128 --finetune --save-label 2 3 4 5
wait
echo "结束训练......"









