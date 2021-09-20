#!/usr/bin/env zsh

# ST_el_li7，增大类间距，减小类内距，李博修正后
# 不再使用 exp，直接算类内 & 类间相似度，类间取绝对值，控制为 0；类内[-1,1]，都减去 1 取值范围使[-2,0]，最后 loss 计算添加负号
echo "开始训练 4......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_li7  --dataset CIFAR10 --gpu-id 3 --seed 1 --list_aug 3 5 --alpha 0.01 --tmp 1
wait
echo "开始训练 5......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_li7  --dataset CIFAR10 --gpu-id 3 --seed 2 --list_aug 3 5 --alpha 0.1 --tmp 1
wait
echo "开始训练 6......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_li7  --dataset CIFAR10 --gpu-id 3 --seed 3 --list_aug 3 5 --alpha 1 --tmp 1
wait
echo "开始训练 7......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_li7  --dataset CIFAR10 --gpu-id 3 --seed 4 --list_aug 3 5 --alpha 5 --tmp 1
wait
echo "开始训练 7......"
CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --model preactresnet --AT-method ST_el_li7  --dataset CIFAR10 --gpu-id 3 --seed 5 --list_aug 3 5 --alpha 10 --tmp 1
wait
echo "结束训练......"





