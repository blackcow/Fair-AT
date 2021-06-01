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
#echo "开始训练 1......"
#python train_trades_cifar10.py --batch-size 192 --gpu-id 0,1,2 --epsilon 0.0078
#wait
#echo "开始训练 2......"
#python train_trades_cifar10.py --batch-size 192 --gpu-id 0,1,2 --epsilon 0.0039
#echo "结束训练......"

#Fair ST（124）
#echo "开始训练 1......"
#python train_trades_cifar10.py --fair v1 --AT-method ST --batch-size 128 --T 0.1
#wait
#echo "开始训练 2......"
#python train_trades_cifar10.py --fair v1 --AT-method ST --batch-size 128 --T 0.3
#wait
#echo "开始训练 3......"
#python train_trades_cifar10.py --fair v1 --AT-method ST --batch-size 128 --T 0.5
#wait
#echo "开始训练 4......"
#python train_trades_cifar10.py --fair v1 --AT-method ST --batch-size 128 --T 0.8
#echo "结束训练......"

#Fair ST（248）
echo "开始训练 1......"
python train_trades_cifar10.py --fair v1a --AT-method ST --batch-size 256 --T 0.1 --lamda 1
wait
echo "开始训练 2......"
python train_trades_cifar10.py --fair v1a --AT-method ST --batch-size 256 --T 0.3 --lamda 1
wait
echo "开始训练 3......"
python train_trades_cifar10.py --fair v1a --AT-method ST --batch-size 256 --T 0.5 --lamda 1
wait
echo "开始训练 4......"
python train_trades_cifar10.py --fair v1a --AT-method ST --batch-size 256 --T 0.8 --lamda 1 --epoch 500 --save-freq 100
wait
echo "结束训练......"

# train_trades_cifar10.py --fair v1a --AT-method TRADES --batch-size 10 --T 0.8 --lamda 1
# Fair AT
#python train_trades_cifar10.py --fair v1a --AT-method TRADES --batch-size 128 --T 0.1 --lamda 1
#wait





