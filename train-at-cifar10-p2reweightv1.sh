#!/usr/bin/env zsh

# AT_p2_reweightv1
# 只对 boundary loss 调整权重，使用的 weight 基于 benign acc 和 mean 的差值

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_p2_reweightv1 --dataset CIFAR10 --gpu-id 1 --seed 1 --reweight 0.05  --discrepancy 0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_p2_reweightv1 --dataset CIFAR10 --gpu-id 1 --seed 2 --reweight 0.05  --discrepancy 3
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_p2_reweightv1 --dataset CIFAR10 --gpu-id 1 --seed 3 --reweight 0.05  --discrepancy 5
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_p2_reweightv1 --dataset CIFAR10 --gpu-id 1 --seed 4 --reweight 0.03  --discrepancy 0
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_p2_reweightv1 --dataset CIFAR10 --gpu-id 1 --seed 5 --reweight 0.03  --discrepancy 3
wait
echo "开始训练 6......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_p2_reweightv1 --dataset CIFAR10 --gpu-id 1 --seed 6 --reweight 0.03  --discrepancy 5
wait
echo "结束训练......"


