#!/usr/bin/env zsh

# 根据 teAT_reweight 的结果做 Reweight
# reweight 每次权重更新的 step_size; discrepancy： 各 label 同 avg 差异大于 discrepancy 便调整权重

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweight --dataset CIFAR10 --gpu-id 1 --seed 1 --reweight 0.05  --discrepancy 0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweight  --dataset CIFAR10 --gpu-id 1 --seed 2 --reweight 0.05  --discrepancy 5
wait
#echo "开始训练 3......"
#CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweight  --dataset CIFAR10 --gpu-id 0 --seed 3 --reweight 0.1  --discrepancy 3
#wait
#echo "开始训练 4......"
#CUDA_VISIBLE_DEVICES=2 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweight  --dataset CIFAR10 --gpu-id 2 --seed 4 --reweight 1  --discrepancy 3
#wait
#echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweight  --dataset CIFAR10 --gpu-id 3 --seed 5 --reweight 5  --discrepancy 3
wait
#echo "开始训练 6......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweight  --dataset CIFAR10 --gpu-id 1 --seed 6 --reweight 1  --discrepancy 0
#wait
#echo "开始训练 7......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweight  --dataset CIFAR10 --gpu-id 1 --seed 7 --reweight 5  --discrepancy 0
#wait
#echo "开始训练 8......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweight  --dataset CIFAR10 --gpu-id 1 --seed 8 --reweight 10  --discrepancy 0
#wait
#echo "开始训练 8......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method AT_reweight  --dataset CIFAR10 --gpu-id 1 --seed 9 --reweight 10  --discrepancy 1
#wait
echo "结束训练......"


