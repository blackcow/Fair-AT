#!/usr/bin/env zsh

# ST remove label
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 1
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 6
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 7
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 8
wait
echo "开始训练 6......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 9
wait
echo "结束训练......"





