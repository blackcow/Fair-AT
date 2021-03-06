#!/usr/bin/env zsh

# TRADES (SVHN)
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES --dataset SVHN --gpu-id 2,3 --seed 1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES --dataset SVHN --gpu-id 2,3 --seed 2
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES --dataset SVHN --gpu-id 2,3 --seed 3
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES --dataset SVHN --gpu-id 2,3 --seed 4
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES --dataset SVHN --gpu-id 2,3 --seed 5
wait
echo "结束训练......"