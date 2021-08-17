#!/usr/bin/env zsh

# TRADES (ImageNet10), rmlabel;先把 0-9 依次训练完
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel 0
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel 1
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel 2
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel 3
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel 4
wait
echo "开始训练 6......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel 5
wait
echo "开始训练 7......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel 6
wait
echo "开始训练 8......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel 7
wait
echo "开始训练 9......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel 8
wait
echo "开始训练 10......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 1 --rmlabel 9
wait
echo "结束训练......"

#echo "开始训练 1......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel 0
#wait
#echo "开始训练 2......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel 1
#wait
#echo "开始训练 3......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel 2
#wait
#echo "开始训练 4......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel 3
#wait
#echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel 4
#wait
#echo "开始训练 6......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel 5
#wait
#echo "开始训练 7......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel 6
#wait
#echo "开始训练 8......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel 7
#wait
#echo "开始训练 9......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel 8
#wait
#echo "开始训练 10......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 2 --rmlabel 9
#wait
#echo "结束训练......"
#echo "开始训练 1......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel 0
#wait
#echo "开始训练 2......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel 1
#wait
#echo "开始训练 3......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel 2
#wait
#echo "开始训练 4......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel 3
#wait
#echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel 4
#wait
#echo "开始训练 6......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel 5
#wait
#echo "开始训练 7......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel 6
#wait
#echo "开始训练 8......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel 7
#wait
#echo "开始训练 9......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel 8
#wait
#echo "开始训练 10......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 3 --rmlabel 9
#wait
#echo "结束训练......"
#
#echo "开始训练 1......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel 0
#wait
#echo "开始训练 2......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel 1
#wait
#echo "开始训练 3......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel 2
#wait
#echo "开始训练 4......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel 3
#wait
#echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel 4
#wait
#echo "开始训练 6......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel 5
#wait
#echo "开始训练 7......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel 6
#wait
#echo "开始训练 8......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel 7
#wait
#echo "开始训练 9......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel 8
#wait
#echo "开始训练 10......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 4 --rmlabel 9
#wait
#echo "结束训练......"
#
#echo "开始训练 1......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel 0
#wait
#echo "开始训练 2......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel 1
#wait
#echo "开始训练 3......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel 2
#wait
#echo "开始训练 4......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel 3
#wait
#echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel 4
#wait
#echo "开始训练 6......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel 5
#wait
#echo "开始训练 7......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel 6
#wait
#echo "开始训练 8......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel 7
#wait
#echo "开始训练 9......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel 8
#wait
#echo "开始训练 10......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10-rmlabel.py --model preactresnet --AT-method TRADES --dataset ImageNet10 --gpu-id 1,2,3 --seed 5 --rmlabel 9
#wait
#echo "结束训练......"


