#!/usr/bin/env zsh

# AT_reweightv1
# 对 natural loss 和 boundary loss 都调整权重，使用相同的 weight（基于 benign acc 和 mean 的差值）

# AT_reweightv1 FashionMNIST
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweightv1 --dataset FashionMNIST --gpu-id 1 --seed 1 --reweight 0.05  --discrepancy 0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweightv1 --dataset FashionMNIST --gpu-id 1 --seed 2 --reweight 0.05  --discrepancy 3
wait
# AT_reweightv1 STL10
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweightv1 --dataset STL10 --gpu-id 1 --seed 1 --reweight 0.05  --discrepancy 0
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweightv1 --dataset STL10 --gpu-id 1 --seed 2 --reweight 0.05  --discrepancy 3
wait
echo "结束训练......"