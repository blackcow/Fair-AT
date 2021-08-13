#!/usr/bin/env zsh

# TRADES_aug_pgdattk2 (CIFAR10)
# 在每个 seed 下使用不同的超参数

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk2 --dataset CIFAR10 --gpu-id 0 --seed 1 --beta_aug 0.5
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk2  --dataset CIFAR10 --gpu-id 0 --seed 2 --beta_aug 1
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk2  --dataset CIFAR10 --gpu-id 0 --seed 3 --beta_aug 2
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk2  --dataset CIFAR10 --gpu-id 0 --seed 4 --beta_aug 4
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk2  --dataset CIFAR10 --gpu-id 0 --seed 5 --beta_aug 6
wait
echo "结束训练......"