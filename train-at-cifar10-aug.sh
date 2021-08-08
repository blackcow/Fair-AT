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

# TRADES_aug_pgd (CIFAR10)
# 在每个 seed 下使用不同的超参数
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgd --dataset CIFAR10 --gpu-id 3 --seed 1 --beta_aug 0.5
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=2 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgd --dataset CIFAR10 --gpu-id 2 --seed 2 --beta_aug 1
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgd --dataset CIFAR10 --gpu-id 1 --seed 3 --beta_aug 2
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgd --dataset CIFAR10 --gpu-id 0 --seed 4 --beta_aug 4
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1,2,3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgd --dataset CIFAR10 --gpu-id 1,2,3 --seed 5 --beta_aug 6
wait
echo "结束训练......"

# TRADES_aug_pgdattk (CIFAR10)
# 在每个 seed 下使用不同的超参数
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk --dataset CIFAR10 --gpu-id 3 --seed 1 --beta_aug 0.5
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=2 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk  --dataset CIFAR10 --gpu-id 2 --seed 2 --beta_aug 1
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk  --dataset CIFAR10 --gpu-id 1 --seed 3 --beta_aug 2
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk  --dataset CIFAR10 --gpu-id 0 --seed 4 --beta_aug 4
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_aug_pgdattk  --dataset CIFAR10 --gpu-id 1 --seed 5 --beta_aug 6
wait
echo "结束训练......"


# TRADES_aug_pgdattk (CIFAR10)
# 在每个 seed 下使用不同的超参数
echo "开始训练 1......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 0 --seed 1 --beta 0.5 --beta_aug 6
wait
echo "开始训练 2......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 0 --seed 2 --beta 1 --beta_aug 6
wait
echo "开始训练 3......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 0 --seed 3 --beta 1 --beta_aug 12
wait
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 0 --seed 4 --beta 1 --beta_aug 18
wait
echo "开始训练 5......"
#CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method TRADES_loss_adp  --dataset CIFAR10 --gpu-id 1 --seed 5 --beta 1 --beta_aug 18
wait
echo "结束训练......"