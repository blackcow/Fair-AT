#!/usr/bin/env zsh

# TRADES_el_li2 (CIFAR10)
# TRADES + ST_el_li2
# 在每个 seed 下使用不同的超参数

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_el_li2  --dataset CIFAR10 --gpu-id 1 --seed 1 --alpha 0.1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_el_li2  --dataset CIFAR10 --gpu-id 1 --seed 2 --alpha 0.5
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_el_li2  --dataset CIFAR10 --gpu-id 1 --seed 3 --alpha 1
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_el_li2  --dataset CIFAR10 --gpu-id 1 --seed 4 --alpha 5
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_el_li2  --dataset CIFAR10 --gpu-id 1 --seed 5 --alpha 10
wait
echo "结束训练......"