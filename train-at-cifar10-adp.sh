#!/usr/bin/env zsh

# TRADES_ST_loss_adp 对特定 label 调整 weight
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_ST_adp  --dataset CIFAR10 --gpu-id 3 --seed 1 --beta 1 --beta_aug 6 --alpha 1.5 --list_aug 2 3 4 5
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_ST_adp  --dataset CIFAR10 --gpu-id 3 --seed 2 --beta 1 --beta_aug 6 --alpha 2 --list_aug 2 3 4 5
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_ST_adp  --dataset CIFAR10 --gpu-id 3 --seed 3 --beta 1 --beta_aug 6 --alpha 3 --list_aug 2 3 4 5
wait
#echo "开始训练 4......"
#CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_ST_adp  --dataset CIFAR10 --gpu-id 3 --seed 1 --beta 1 --beta_aug 6 --alpha 0.1 --list_aug 2 3 4 5
#wait
#echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_ST_adp  --dataset CIFAR10 --gpu-id 3 --seed 1 --beta 1 --beta_aug 6 --alpha 0.1 --list_aug 2 3 4 5
#wait
echo "结束训练......"