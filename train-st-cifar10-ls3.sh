#!/usr/bin/env zsh

# st_ls35，对【2，3，4，5】做 label smooth，其他不做
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_label_smooth25  --dataset CIFAR10 --gpu-id 3 --seed 1 --smooth 0.05
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_label_smooth25  --dataset CIFAR10 --gpu-id 3 --seed 2 --smooth 0.1
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_label_smooth25  --dataset CIFAR10 --gpu-id 3 --seed 3 --smooth 0.4
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_label_smooth25  --dataset CIFAR10 --gpu-id 3 --seed 4 --smooth 0.6
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_label_smooth25  --dataset CIFAR10 --gpu-id 3 --seed 5 --smooth 0.8
wait
echo "结束训练......"





