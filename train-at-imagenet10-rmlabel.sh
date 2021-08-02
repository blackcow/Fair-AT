#!/usr/bin/env zsh

# TRADES (ImageNet10), rmlabel
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel0
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel0
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel0
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel0
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel1
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel1
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel1
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel1
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel2
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel2
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel2
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel2
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel2
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel3
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel3
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel3
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel3
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel3
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel4
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel4
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel4
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel4
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel4
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel5
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel5
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel5
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel5
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel5
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel6
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel6
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel6
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel6
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel6
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel7
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel7
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel7
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel7
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel7
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel8
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel8
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel8
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel8
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel8
wait
echo "结束训练......"

echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel9
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel9
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel9
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel9
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel9
wait
echo "结束训练......"