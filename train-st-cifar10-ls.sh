#!/usr/bin/env zsh

# ST_label_smooth,使用 label smooth ce loss 用于全部 label data
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_label_smooth  --dataset CIFAR10 --gpu-id 3 --seed 1 --smooth 0.05
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_label_smooth  --dataset CIFAR10 --gpu-id 3 --seed 2 --smooth 0.1
wait
echo "开始训练 6......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_label_smooth  --dataset CIFAR10 --gpu-id 3 --seed 3 --smooth 0.2
wait
echo "开始训练 7......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_label_smooth  --dataset CIFAR10 --gpu-id 3 --seed 4 --smooth 0.3
wait
echo "开始训练 7......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_label_smooth  --dataset CIFAR10 --gpu-id 3 --seed 5 --smooth 0.5
wait
echo "结束训练......"