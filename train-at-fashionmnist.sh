#!/usr/bin/env zsh

# AT FashionMNIST
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES --dataset FashionMNIST --gpu-id 2,3 --seed 1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES --dataset FashionMNIST --gpu-id 2,3 --seed 2
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES --dataset FashionMNIST --gpu-id 2,3 --seed 3
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES --dataset FashionMNIST --gpu-id 2,3 --seed 4
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=0,1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES --dataset FashionMNIST --gpu-id 2,3 --seed 5
wait
echo "结束训练......"