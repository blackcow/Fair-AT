#!/usr/bin/env zsh

# st_el_logits，只计算 3-5 在 logits 上的内积，让值尽量小，增大类间距
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_logits  --dataset CIFAR10 --gpu-id 0 --seed 1 --list_aug 3 5 --alpha 0.1 --tmp 1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_logits  --dataset CIFAR10 --gpu-id 0 --seed 2 --list_aug 3 5 --alpha 0.5 --tmp 1
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_logits  --dataset CIFAR10 --gpu-id 0 --seed 3 --list_aug 3 5 --alpha 1 --tmp 1
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_logits  --dataset CIFAR10 --gpu-id 0 --seed 4 --list_aug 3 5 --alpha 2 --tmp 1
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_logits  --dataset CIFAR10 --gpu-id 0 --seed 5 --list_aug 3 5 --alpha 4 --tmp 1
wait
echo "结束训练......"



