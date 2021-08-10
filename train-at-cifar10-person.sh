#!/usr/bin/env zsh

# TRADES_aug (CIFAR10)
# 在每个 seed 下使用不同的超参数
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 0 --percent 0.2
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 0 --percent 0.4
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 0 --percent 0.6
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 0 --percent 0.8
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 0 --percent 1
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 1 --percent 0.2
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 1 --percent 0.4
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 1 --percent 0.6
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 1 --percent 0.8
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 1 --percent 1
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 2 --percent 0.2
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 2 --percent 0.4
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 2 --percent 0.6
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 2 --percent 0.8
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 2 --percent 1
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 3 --percent 0.2
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 3 --percent 0.4
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 3 --percent 0.6
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 3 --percent 0.8
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 3 --percent 1
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 4 --percent 0.2
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 4 --percent 0.4
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 4 --percent 0.6
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 4 --percent 0.8
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 4 --percent 1
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 5 --percent 0.2
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 5 --percent 0.4
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 5 --percent 0.6
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 5 --percent 0.8
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES  --dataset CIFAR10 --gpu-id 0 --seed 1 --rmlabel 5 --percent 1
wait
echo "结束训练......"