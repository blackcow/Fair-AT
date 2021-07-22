#!/usr/bin/env zsh

# ST keep percent data(CIFAR-100)
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 0.1 --model preactresnet --seed 1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 0.1 --model preactresnet --seed 2
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 0.1 --model preactresnet --seed 3
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 0.1 --model preactresnet --seed 4
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 0.1 --model preactresnet --seed 5
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 0.5 --model preactresnet --seed 1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 0.5 --model preactresnet --seed 2
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 0.5 --model preactresnet --seed 3
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 0.5 --model preactresnet --seed 4
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 0.5 --model preactresnet --seed 5
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 1 --model preactresnet --seed 1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 1 --model preactresnet --seed 2
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 1 --model preactresnet --seed 3
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 1 --model preactresnet --seed 4
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1 python train_st_cifar10_partlabel.py --dataset CIFAR100 --percent 1 --model preactresnet --seed 5
wait
echo "结束训练......"




