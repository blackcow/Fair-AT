import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


# 返回更新的中心点，总计数
def update(rep_center, rep_temp, rep_num, batch_num):
    rep_center = rep_num / (rep_num + batch_num) * rep_center + rep_temp.sum() / (rep_num + batch_num)
    return rep_center


# v1-v3 使用的 fair loss，input 同 label 中心近
def FairLoss1(args, rep, rep_center, target):
    # 归一化，计算 input 同 rep_center 计算余弦相似度
    rep = nn.functional.normalize(rep, dim=1)
    rep_center = nn.functional.normalize(rep_center, dim=1)
    logits = torch.einsum('nm,km->nk', [rep, rep_center.clone().detach()])  # logits: [N, K]
    # torch.mm(rep, torch.transpose(rep_center, 0, 1))
    # apply temperature
    logits /= args.T

    # labels: positive key indicators
    fair_loss = F.cross_entropy(logits, target)
    return fair_loss

# 计算 rep 中心，然后同最近邻的 rep 中心，尽量远
def FairLoss2(args, rep, rep_center, target):
    # [10, H*W]
    rep_center = nn.functional.normalize(rep_center, dim=1)
    logits = torch.mm(rep_center.clone().detach(), torch.transpose(rep_center, 0, 1).clone().detach())  # [10,HW]*[HW,10]=[10,10]
    logits = logits - torch.diag_embed(torch.diag(logits))  # 去掉对角线的 1
    fair_loss = logits.max(0)
    fair_loss = fair_loss[0].abs().sum()
    # sim = F.cosine_similarity(rep, rep.clone().detach())
    return fair_loss
    # return torch.ones(1).requires_grad_(True)

# fl2：最近的，尽量远
# fl3：（最）远的，尽量近（让 rep 中心，尽量互相混淆）
def FairLoss3(args, rep, rep_center, target):
    # [10, H*W]
    rep_center = nn.functional.normalize(rep_center, dim=1)
    logits = torch.mm(rep_center.clone().detach(), torch.transpose(rep_center, 0, 1).clone().detach())  # [10,HW]*[HW,10]=[10,10]
    # logits = logits - torch.ones_like(logits)
    fair_loss = (torch.ones_like(logits) - logits).sum()  # 逼近 1，相互混淆
    return fair_loss

# class FairLoss2(nn.Module):
#     def __init__(self, lamda):
#         super(FairLoss2, self).__init__()
#         self.lamda = lamda
#
#     def forward(self, rep):
#         # [10, H*W]
#         logits = torch.mm(rep, torch.transpose(rep, 0, 1))  # [10,HW]*[HW,10]=[10,10]
#         logits = logits - torch.diag_embed(torch.diag(logits))  # 去掉对角线的 1
#         logits = logits.abs().sum()
#         # sim = F.cosine_similarity(rep, rep.clone().detach())
#         return logits * self.lamda
#         # return torch.ones(1).requires_grad_(True)

def fair_loss(args, target, rep_center, rep, out):
    # rep, out = model(x)
    # 得到 input 的 rep，归一化并展开
    N, C, H, W = rep[0].size()
    rep[0] = rep[0].reshape([N, -1])  # [N,M] [128,40960]
    rep[1] = rep[1].reshape([N, -1])  # [N,M] [128,40960]

    # 只考虑 batch 内部
    target_tmp = target.cpu().numpy()
    for i in range(10):
        # 获取 label i 数据的索引，找到对应的 rep
        index = np.squeeze(np.argwhere(target_tmp == i))
        index1 = torch.tensor(index).cuda()
        rep_benign_temp = torch.index_select(rep[0], 0, index1)
        rep_robust_temp = torch.index_select(rep[1], 0, index1)

        # 更新 label i 的中心, rep_center [10, 40960]
        # fair v1：新的中心点，占 50%的权重；如果该 batch 中没有样本，则变成原来的一半
        # 根据 batch 计算中心
        if args.fair == 'v1':
            rep_center[0][i] = (rep_center[0][i] + rep_benign_temp.mean(dim=0)) / 2
            rep_center[1][i] = (rep_center[1][i] + rep_robust_temp.mean(dim=0)) / 2
        # 当前 batch 的 data 均值，作为中心点
        elif args.fair == 'v1a':
            rep_center[0][i] = rep_benign_temp.mean(dim=0)
            rep_center[1][i] = rep_robust_temp.mean(dim=0)

        # fair v2：最终每个样本，占中心点的 1/n 的权重
        # elif args.fair == 'v2':
        #     batch_num, _ = rep_temp.size()
        #     rep_center[i] = update(rep_center[i], rep_temp, rep_num[i], batch_num)  # 更新中心点
        #     rep_num[i] += batch_num
        #
        #     CEloss = F.cross_entropy(out, target)
        #     loss = CEloss + args.lamda * FairLoss1(rep, rep_center, target, args)

        # 同 BN 一致，之前的占 90%，新的占 10%
        elif args.fair == 'v3':
            rep_center[0][i] = rep_center[0][i] * 0.9 + rep_benign_temp.mean(dim=0) * 0.1
            rep_center[1][i] = rep_center[1][i] * 0.9 + rep_robust_temp.mean(dim=0) * 0.1

    CEloss = F.cross_entropy(out, target)
    # loss = CEloss + args.lamda * FairLoss1(args, rep=rep[0], rep_center=rep_center[0], target=target) \
    #        + args.lamda * FairLoss1(args, rep=rep[1], rep_center=rep_center[1], target=target)
    # loss = CEloss + args.lamda * FairLoss2(args, rep=rep[0], rep_center=rep_center[0], target=target) \
    #        + args.lamda * FairLoss2(args, rep=rep[1], rep_center=rep_center[1], target=target)
    if args.fairloss == 'fl3':
        loss = CEloss + args.fl_lamda * FairLoss3(args, rep=rep[0], rep_center=rep_center[0], target=target) \
               + args.fl_lamda * FairLoss3(args, rep=rep[1], rep_center=rep_center[1], target=target)
    elif args.fairloss == 'fl2':
        loss = CEloss + args.lamda * FairLoss2(args, rep=rep[0], rep_center=rep_center[0], target=target) \
               + args.lamda * FairLoss2(args, rep=rep[1], rep_center=rep_center[1], target=target)
    elif args.fairloss == 'fl1':
        loss = CEloss + args.lamda * FairLoss1(args, rep=rep[0], rep_center=rep_center[0], target=target) \
               + args.lamda * FairLoss1(args, rep=rep[1], rep_center=rep_center[1], target=target)
    else:
        raise ValueError('no fairloss!')
    # # 只看 label 中心点之间的距离，作为 loss
    # if args.fair == 'v4':  # 目前 rep 的距离看来，没达到与其效果
    #     rep_center[i] = rep_center[i] * 0.9 + rep_temp.mean(dim=0) * 0.1
    #     # 归一化，计算 input 同 rep_center 计算余弦相似度
    #     rep_center = rep_center.detach()
    #     rep_center_norm = nn.functional.normalize(rep_center, dim=1)
    #
    #     # 针对 label 中心互相远离的 loss
    #     fl = FairLoss2(args.lamda)
    #     CEloss = F.cross_entropy(out, target)
    #     loss = CEloss + fl(rep_center_norm)
    return rep_center, loss
