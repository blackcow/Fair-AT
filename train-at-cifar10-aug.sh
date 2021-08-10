#!/usr/bin/env zsh

# TRADES_aug (CIFAR10)
# 在每个 seed 下使用不同的超参数
#echo "开始训练 1......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug --dataset CIFAR10 --gpu-id 0 --seed 1 --beta_aug 6
#wait
#echo "开始训练 2......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug --dataset CIFAR10 --gpu-id 0 --seed 2 --beta_aug 8
#wait
#echo "开始训练 3......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug --dataset CIFAR10 --gpu-id 0 --seed 3 --beta_aug 10
#wait
#echo "开始训练 4......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug --dataset CIFAR10 --gpu-id 0 --seed 4 --beta_aug 12
#wait
#echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug --dataset CIFAR10 --gpu-id 0 --seed 5 --beta_aug 18
#wait
#echo "结束训练......"
#
## TRADES_aug_pgd (CIFAR10)
## 在每个 seed 下使用不同的超参数
#echo "开始训练 1......"
#CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgd --dataset CIFAR10 --gpu-id 3 --seed 1 --beta_aug 0.5
#wait
#echo "开始训练 2......"
#CUDA_VISIBLE_DEVICES=2 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgd --dataset CIFAR10 --gpu-id 2 --seed 2 --beta_aug 1
#wait
#echo "开始训练 3......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgd --dataset CIFAR10 --gpu-id 1 --seed 3 --beta_aug 2
#wait
#echo "开始训练 4......"
#CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgd --dataset CIFAR10 --gpu-id 0 --seed 4 --beta_aug 4
#wait
#echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgd --dataset CIFAR10 --gpu-id 1,2,3 --seed 5 --beta_aug 6
#wait
#echo "结束训练......"
#
## TRADES_aug_pgdattk (CIFAR10)，使用 PGD 对 2-5 label 额外生成 adv 样本
## 在每个 seed 下使用不同的超参数
#echo "开始训练 1......"
#CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk --dataset CIFAR10 --gpu-id 3 --seed 1 --beta_aug 0.5
#wait
#echo "开始训练 2......"
#CUDA_VISIBLE_DEVICES=2 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk  --dataset CIFAR10 --gpu-id 2 --seed 2 --beta_aug 1
#wait
#echo "开始训练 3......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk  --dataset CIFAR10 --gpu-id 1 --seed 3 --beta_aug 2
#wait
#echo "开始训练 4......"
#CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk  --dataset CIFAR10 --gpu-id 0 --seed 4 --beta_aug 4
#wait
#echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk  --dataset CIFAR10 --gpu-id 1 --seed 5 --beta_aug 6
#wait
#echo "结束训练......"


# TRADES_loss_adp (CIFAR10),Trade loss 根据 label 不同权重不一致
# 在每个 seed 下使用不同的超参数
#echo "开始训练 1......"
#CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 0 --seed 1 --beta 0.1 --beta_aug 6 --list_aug 2 3 4 5
#wait
#echo "开始训练 2......"
#CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 0 --seed 2 --beta 0.5 --beta_aug 6 --list_aug 2 3 4 5
#wait
#echo "开始训练 3......"
#CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 0 --seed 3 --beta 1 --beta_aug 6 --list_aug 2 3 4 5
#wait
#echo "开始训练 4......"
#CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 0 --seed 4 --beta 1 --beta_aug 12 --list_aug 2 3 4 5
#wait
#echo "开始训练 5......"
##CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 0 --seed 5 --beta 6 --beta_aug 18 --list_aug 2 3 4 5
##CUDA_VISIBLE_DEVICES=2 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 2 --seed 6 --beta 4 --beta_aug 12 --list_aug 2 3 4 5
#wait
## 只对 3 做 aug；3，5 一起 aug
##CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 0 --seed 7 --beta 1 --beta_aug 6 --list_aug 3
##CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 0 --seed 8 --beta 1 --beta_aug 6 --list_aug 3 5
#echo "结束训练......"

# TRADES_aug (CIFAR10)
# 在每个 seed 下使用不同的超参数
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=2 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_augmulti --dataset CIFAR10 --gpu-id 2 --seed 1 --beta_aug 1
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_augmulti --dataset CIFAR10 --gpu-id 3 --seed 2 --beta_aug 2
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_augmulti --dataset CIFAR10 --gpu-id 1 --seed 3 --beta_aug 4
wait
echo "开始训练 4......"(还没训练)
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_augmulti --dataset CIFAR10 --gpu-id 0 --seed 4 --beta_aug 6
wait
echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_augmulti --dataset CIFAR10 --gpu-id 0 --seed 5 --beta_aug 18
wait
echo "结束训练......"