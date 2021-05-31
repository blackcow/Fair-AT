import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


# 返回更新的中心点，总计数
def update(rep_center, rep_temp, rep_num, batch_num):
    # rep_center = (rep_center * rep_num + rep_temp.sum())/(rep_num+batch_num)
    rep_center = rep_num / (rep_num + batch_num) * rep_center + rep_temp.sum() / (rep_num + batch_num)
    return rep_center


# v1-v3 使用的 fair loss，input 同 label 中心近
def FairLoss1(rep, rep_center, target, args):
    # 归一化，计算 input 同 rep_center 计算余弦相似度
    rep = nn.functional.normalize(rep, dim=1)
    rep_center = nn.functional.normalize(rep_center, dim=1)
    logits = torch.einsum('nm,km->nk', [rep, rep_center.clone().detach()])  # logits: [N, K]

    # apply temperature
    logits /= args.T

    # labels: positive key indicators
    fair_loss = F.cross_entropy(logits, target)
    return fair_loss

class FairLoss2(nn.Module):
    def __init__(self, lamda):
        super(FairLoss2, self).__init__()
        self.lamda = lamda

    def forward(self, rep):
        # [10, H*W]
        logits = torch.mm(rep, torch.transpose(rep, 0, 1))  # [10,HW]*[HW,10]=[10,10]
        logits = logits - torch.diag_embed(torch.diag(logits))  # 去掉对角线的 1
        logits = logits.abs().sum()
        # sim = F.cosine_similarity(rep, rep.clone().detach())
        return logits * self.lamda
        # return torch.ones(1).requires_grad_(True)

def fair_loss(model, x, target, args, rep_center, fair):
    rep, out = model(x)
    loss = F.cross_entropy(out, target)
    # 得到 input 的 rep，归一化并展开
    N, C, H, W = rep.size()
    rep = rep.reshape([N, -1])  # [N,M] [128,40960]

    # 只考虑 batch 内部
    target_tmp = target.cpu().numpy()
    for i in range(10):
        # 获取 label i 数据的索引，找到对应的 rep
        index = np.squeeze(np.argwhere(target_tmp == i))
        index1 = torch.tensor(index).cuda()
        rep_temp = torch.index_select(rep, 0, index1)

        # 更新 label i 的中心, rep_center [10, 40960]
        # fair v1：新的中心点，占 50%的权重；如果该 batch 中没有样本，则变成原来的一半
        if args.fair == 'v1':
            rep_center[i] = (rep_center[i] + rep_temp.mean(dim=0)) / 2

            CEloss = F.cross_entropy(out, target)
            loss = CEloss + args.lamda * FairLoss1(rep, rep_center, target, args)
        # 当前 batch 的 data 均值，作为中心点
        if args.fair == 'v1a':
            rep_center[i] = rep_temp.mean(dim=0)

            CEloss = F.cross_entropy(out, target)
            loss = CEloss + args.lamda * FairLoss1(rep, rep_center, target, args)

        # fair v2：最终每个样本，占中心点的 1/n 的权重
        # if args.fair == 'v2':
        #     batch_num, _ = rep_temp.size()
        #     rep_center[i] = update(rep_center[i], rep_temp, rep_num[i], batch_num)  # 更新中心点
        #     rep_num[i] += batch_num
        #
        #     CEloss = F.cross_entropy(out, target)
        #     loss = CEloss + args.lamda * FairLoss1(rep, rep_center, target, args)

        # 同 BN 一致，之前的占 90%，新的占 10%
        if args.fair == 'v3':
            rep_center[i] = rep_center[i] * 0.9 + rep_temp.mean(dim=0) * 0.1

            CEloss = F.cross_entropy(out, target)
            loss = CEloss + args.lamda * FairLoss1(rep, rep_center, target, args)

        # 只看 label 中心点之间的距离，作为 loss
        if args.fair == 'v4':  # 目前 rep 的距离看来，没达到与其效果
            rep_center[i] = rep_center[i] * 0.9 + rep_temp.mean(dim=0) * 0.1
            # 归一化，计算 input 同 rep_center 计算余弦相似度
            rep_center = rep_center.detach()
            rep_center_norm = nn.functional.normalize(rep_center, dim=1)

            # 针对 label 中心互相远离的 loss
            fl = FairLoss2(args.lamda)
            CEloss = F.cross_entropy(out, target)
            loss = CEloss + fl(rep_center_norm)
    return rep_center, loss
