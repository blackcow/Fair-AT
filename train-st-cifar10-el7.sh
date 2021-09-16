#!/usr/bin/env zsh

# ST_el_li3，增大类间距，减小类内距，李博修正后
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_li3  --dataset CIFAR10 --gpu-id 3 --seed 1 --list_aug 3 5 --alpha 0.01 --tmp 1
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_li3  --dataset CIFAR10 --gpu-id 3 --seed 2 --list_aug 3 5 --alpha 0.1 --tmp 1
wait
echo "开始训练 6......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_li3  --dataset CIFAR10 --gpu-id 3 --seed 3 --list_aug 3 5 --alpha 1 --tmp 1
wait
echo "开始训练 7......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_li3  --dataset CIFAR10 --gpu-id 3 --seed 4 --list_aug 3 5 --alpha 5 --tmp 1
wait
echo "开始训练 7......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_li3  --dataset CIFAR10 --gpu-id 3 --seed 5 --list_aug 3 5 --alpha 10 --tmp 1
wait
echo "结束训练......"





