#!/usr/bin/env zsh

# AT remove label(0-9)，使用 seed3 做
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 0 --seed 3
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 1 --seed 3
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 2 --seed 3
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 3 --seed 3
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 4 --seed 3
wait
echo "开始训练 6......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 5 --seed 3
wait
echo "开始训练 7......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 6 --seed 3
wait
echo "开始训练 8......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 7 --seed 3
wait
echo "开始训练 9......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 8 --seed 3
wait
echo "开始训练 10......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10-rmlabel.py  --model preactresnet --AT-method TRADES --batch-size 128 --rmlabel 9 --seed 3
wait
echo "结束训练......"









