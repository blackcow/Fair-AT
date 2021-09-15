#!/usr/bin/env zsh

# AT_reweightv2
# 对 natural loss 和 boundary loss 都调整权重，分别使用不同的 weight

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweightv2 --dataset CIFAR10 --gpu-id 1 --seed 1 --reweight 0.05  --discrepancy 0 --test_attack
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweightv2 --dataset CIFAR10 --gpu-id 1 --seed 2 --reweight 0.05  --discrepancy 3 --test_attack
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweightv2 --dataset CIFAR10 --gpu-id 1 --seed 3 --reweight 0.05  --discrepancy 5 --test_attack
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweightv2 --dataset CIFAR10 --gpu-id 1 --seed 4 --reweight 0.03  --discrepancy 0 --test_attack
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweightv2 --dataset CIFAR10 --gpu-id 1 --seed 5 --reweight 0.03  --discrepancy 3 --test_attack
wait
echo "开始训练 6......"
CUDA_VISIBLE_DEVICES=2,3 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweightv2 --dataset CIFAR10 --gpu-id 1 --seed 6 --reweight 0.03  --discrepancy 5 --test_attack
wait
echo "结束训练......"


