#!/usr/bin/env zsh

# ST remove label
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 5 --percent 0.2 --gpu-id 0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 5 --percent 0.5 --gpu-id 0
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 5 --percent 0.7 --gpu-id 0
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 5 --percent 0.9 --gpu-id 0
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 6 --percent 0.2 --gpu-id 0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 6 --percent 0.5 --gpu-id 0
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 6 --percent 0.7 --gpu-id 0
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 6 --percent 0.9 --gpu-id 0
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 7 --percent 0.2 --gpu-id 0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 7 --percent 0.5 --gpu-id 0
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 7 --percent 0.7 --gpu-id 0
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 7 --percent 0.9 --gpu-id 0
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 8 --percent 0.2 --gpu-id 0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 8 --percent 0.5 --gpu-id 0
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 8 --percent 0.7 --gpu-id 0
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 8 --percent 0.9 --gpu-id 0
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 9 --percent 0.2 --gpu-id 0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 9 --percent 0.5 --gpu-id 0
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 9 --percent 0.7 --gpu-id 0
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_st_cifar10-rmlabel.py --model preactresnet --AT-method ST --batch-size 128 --rmlabel 9 --percent 0.9 --gpu-id 0
wait
echo "结束训练......"




