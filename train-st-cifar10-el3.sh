#!/usr/bin/env zsh

# 在 ST 对特定 label，只 enlarge 类间距，不管类内距离
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_only_el  --dataset CIFAR10 --gpu-id 3 --seed 1 --list_aug 3 5 --alpha 0.1 --tmp 1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_only_el  --dataset CIFAR10 --gpu-id 3 --seed 2 --list_aug 3 5 --alpha 0.5 --tmp 1
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_only_el  --dataset CIFAR10 --gpu-id 3 --seed 3 --list_aug 3 5 --alpha 1 --tmp 1
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_only_el  --dataset CIFAR10 --gpu-id 3 --seed 4 --list_aug 3 5 --alpha 2 --tmp 1
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_only_el  --dataset CIFAR10 --gpu-id 3 --seed 5 --list_aug 3 5 --alpha 4 --tmp 1
wait
echo "结束训练......"





