#!/usr/bin/env zsh

# ST_mixup，所有 label 都 mix
#echo "开始训练 4......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_mixup  --dataset CIFAR10 --gpu-id 3 --seed 1 --mixalpha 1
#wait
#echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_mixup  --dataset CIFAR10 --gpu-id 3 --seed 2 --mixalpha 0.7
#wait
#echo "开始训练 6......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_mixup  --dataset CIFAR10 --gpu-id 3 --seed 3 --mixalpha 0.5
#wait
#echo "开始训练 7......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_mixup  --dataset CIFAR10 --gpu-id 3 --seed 4 --mixalpha 0.3
#wait
#echo "开始训练 7......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_mixup  --dataset CIFAR10 --gpu-id 3 --seed 5 --mixalpha 0.1
#wait
echo "开始训练 6......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_mixup  --dataset CIFAR10 --gpu-id 3 --seed 6 --mixalpha 2
wait
echo "开始训练 7......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_mixup  --dataset CIFAR10 --gpu-id 3 --seed 7 --mixalpha 5
wait
echo "开始训练 8......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_mixup  --dataset CIFAR10 --gpu-id 3 --seed 8 --mixalpha 10
wait
echo "结束训练......"





