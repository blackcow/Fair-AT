#!/usr/bin/env zsh

# 根据 teST_reweight 的结果做 Reweight 
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_reweight --dataset CIFAR10 --gpu-id 1 --seed 1 --reweight 0.05  --discrepancy 0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_reweight  --dataset CIFAR10 --gpu-id 1 --seed 2 --reweight 0.05  --discrepancy 5
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_reweight  --dataset CIFAR10 --gpu-id 1 --seed 3 --reweight 0.1  --discrepancy 3
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_reweight  --dataset CIFAR10 --gpu-id 1 --seed 4 --reweight 1  --discrepancy 3
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_reweight  --dataset CIFAR10 --gpu-id 1 --seed 5 --reweight 5  --discrepancy 3
wait
echo "结束训练......"


