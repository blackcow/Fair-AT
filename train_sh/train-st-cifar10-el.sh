#!/usr/bin/env zsh

# 在 ST 对特定 label，enlarge 类间距，减小类内距离
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el  --dataset CIFAR10 --gpu-id 1 --seed 1 --list_aug 3 5 --alpha 1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el  --dataset CIFAR10 --gpu-id 1 --seed 2 --list_aug 3 5 --alpha 2
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el  --dataset CIFAR10 --gpu-id 1 --seed 3 --list_aug 3 5 --alpha 4
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el  --dataset CIFAR10 --gpu-id 1 --seed 4 --list_aug 3 5 --alpha 10
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el  --dataset CIFAR10 --gpu-id 1 --seed 5 --list_aug 3 5 --alpha 20
wait
echo "结束训练......"

# seed 6  temperatures 由 0.1 调整为 1
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el  --dataset CIFAR10 --gpu-id 1 --seed 6 --list_aug 3 5 --alpha 2 --temperature 1

# 减小 alpha
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method ST_el  --dataset CIFAR10 --gpu-id 1 --seed 7 --list_aug 3 5 --alpha 0.2 --temperature 1