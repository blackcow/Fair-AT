#!/usr/bin/env zsh

# ST_el_fix，增大类间距，减小类内距，李博修正后
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_fix  --dataset CIFAR10 --gpu-id 3 --seed 1 --list_aug 3 5 --alpha 1 --tmp 0.1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_fix  --dataset CIFAR10 --gpu-id 3 --seed 2 --list_aug 3 5 --alpha 5 --tmp 0.1
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_fix  --dataset CIFAR10 --gpu-id 3 --seed 3 --list_aug 3 5 --alpha 10 --tmp 0.1
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_fix  --dataset CIFAR10 --gpu-id 3 --seed 4 --list_aug 3 5 --alpha 0.1 --tmp 1
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_fix  --dataset CIFAR10 --gpu-id 3 --seed 5 --list_aug 3 5 --alpha 0.5 --tmp 1
wait
echo "开始训练 6......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_fix  --dataset CIFAR10 --gpu-id 3 --seed 6 --list_aug 3 5 --alpha 1 --tmp 1
wait
echo "开始训练 7......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_fix  --dataset CIFAR10 --gpu-id 3 --seed 7 --list_aug 3 5 --alpha 5 --tmp 1
wait
echo "结束训练......"





