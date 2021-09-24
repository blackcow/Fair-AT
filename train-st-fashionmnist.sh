#!/usr/bin/env zsh

# ST FashionMNIST
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --dataset FashionMNIST --AT-method ST --model preactresnet --seed 1 --gpu-id 1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --dataset FashionMNIST --AT-method ST --model preactresnet --seed 2 --gpu-id 1
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --dataset FashionMNIST --AT-method ST --model preactresnet --seed 3 --gpu-id 1
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --dataset FashionMNIST --AT-method ST --model preactresnet --seed 4 --gpu-id 1
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --dataset FashionMNIST --AT-method ST --model preactresnet --seed 5 --gpu-id 1
wait
echo "结束训练......"
