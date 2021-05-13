#!/usr/bin/env bash
#echo "开始训练 1......"
#python train_trades_cifar10.py --batch-size 192 --gpu-id 0,1,2 --epsilon 0.0078
#wait
#echo "开始训练 2......"
#python train_trades_cifar10.py --batch-size 192 --gpu-id 0,1,2 --epsilon 0.0039
##wait
##echo "开始训练 3......"
##python /home/yckj2334/sh_batch/python3.py
#echo "结束训练......"



# 137 服务器
#echo "开始训练 1......"
#python train_trades_cifar10.py  --batch-size 192 --gpu-id 2,1,0 --widen_factor 4
#wait
#echo "开始训练 2......"
#python train_trades_cifar10.py  --batch-size 192 --gpu-id 2,1,0 --widen_factor 6
#echo "结束训练......"


# 248 服务器
#echo "开始训练 1......"
#python train_trades_cifar10.py --batch-size 192 --gpu-id 0,1,2 --droprate 0.1
#wait
#echo "开始训练 2......"
#python train_trades_cifar10.py --batch-size 192 --gpu-id 0,1,2 --droprate 0.2
#wait
#echo "开始训练 3......"
#python train_trades_cifar10.py --batch-size 192 --gpu-id 0,1,2 --droprate 0.3
#wait
#echo "开始训练 4......"
#python train_trades_cifar10.py --batch-size 192 --gpu-id 0,1,2 --droprate 0.4
#wait
#echo "开始训练 5......"
#python train_trades_cifar10.py --batch-size 192 --gpu-id 0,1,2 --droprate 0.5
#echo "结束训练......"

# epsilon；248 服务器
echo "开始训练 1......"
python train_trades_cifar10.py --batch-size 192 --gpu-id 0,1,2 --epsilon 0.0078
wait
echo "开始训练 2......"
python train_trades_cifar10.py --batch-size 192 --gpu-id 0,1,2 --epsilon 0.0039
echo "结束训练......"




