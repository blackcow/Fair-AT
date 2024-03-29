import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
# from pgd import attack_pgd
from pgd import *
import numpy as np

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta
            _, logits = model(adv)
            _, logits_x = model(x_natural)
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(logits, dim=1),
                                           F.softmax(logits_x, dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_x, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_x, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

# 只对 ST ，在 conflicting pair 上 enlarge distance
def trades_loss_el(model, x_natural, y, optimizer, list_aug, alpha, temperature,
                   step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta
            _, logits = model(adv)
            _, logits_x = model(x_natural)
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(logits, dim=1),
                                           F.softmax(logits_x, dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    rep_x, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_x, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_x, dim=1))

    # el loss
    # 手动设定 conflict pair
    idx1, idx2 = [], []
    idx1.append((y == 3).nonzero().flatten())
    idx2.append((y == 5).nonzero().flatten())
    idx1 = torch.cat(idx1)
    idx2 = torch.cat(idx2)
    len_1 = len(idx1)
    len_2 = len(idx2)

    # calculate natural loss
    rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
    rep_x = F.normalize(rep_x.squeeze(), dim=1)
    rep_x_p1 = torch.index_select(rep_x, 0, idx1)
    rep_x_p2 = torch.index_select(rep_x, 0, idx2)
    y1 = torch.index_select(y, 0, idx1)
    y2 = torch.index_select(y, 0, idx2)

    # 计算 intra dis，类内距离
    rep_intra1 = torch.matmul(rep_x_p1, rep_x_p1.T) / temperature
    # rep_intra2 = torch.matmul(rep_x_p2, rep_x_p2.T) / temperature
    # Log-sum trick for numerical stability
    # 计算内积距离后，最近的值为 1/ self.temperature，所以减去最大值后，最近为 0，远离为负值
    logits_max1, _ = torch.max(rep_intra1, dim=1, keepdim=True)
    # logits_max2, _ = torch.max(rep_intra2, dim=1, keepdim=True)
    # logits_intra1 = rep_intra1 - logits_max1.detach()
    # logits_intra2 = rep_intra2 - logits_max2.detach()
    # exp_logits1 = torch.exp(logits_intra1)
    # exp_logits2 = torch.exp(logits_intra2)

    # 计算 inter dis
    rep_inter = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
    logits_max, _ = torch.max(rep_inter, dim=1, keepdim=True)
    logits_inter = rep_inter - logits_max1.detach()
    exp_inter = torch.exp(logits_inter)

    # 计算 intra 相似度，对角线的 1 减掉；计算 inter 相似度；两者相除
    # prob = (exp_logits1.sum()-len_1 + exp_logits2.sum()-len_2) / exp_inter.sum()
    prob = 1 / exp_inter.sum()

    # Mean log-likelihood for positive
    # inter loss，类间距离
    loss_el = - (torch.log((prob))) / (len_1 + len_2)


    loss = loss_natural + beta * loss_robust + loss_el * alpha
    return loss


# 针对特定 label ST, 调整 conflict pair 之间 feature 的距离
# [2,3,4,5] ST loss 调整权重，权重改为内部调整
def st_el_li2(model, x_natural, y, list_aug, alpha, temperature):
    # temperature = 0.1
    idx1 = []
    idx2 = []
    idx1.append((y == 3).nonzero().flatten())
    idx2.append((y == 5).nonzero().flatten())
    idx1 = torch.cat(idx1)
    idx2 = torch.cat(idx2)
    len_1 = len(idx1)
    len_2 = len(idx2)

    rep_x, logits_x = model(x_natural)
    loss_natural = F.cross_entropy(logits_x, y)

    if len_1 == 0 or len_2 == 0:
        loss_el = 0
        print(len_1, len_2)
    else:
        rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
        rep_x = F.normalize(rep_x.squeeze(), dim=1)
        rep_x_p1 = torch.index_select(rep_x, 0, idx1)
        rep_x_p2 = torch.index_select(rep_x, 0, idx2)
        y1 = torch.index_select(y, 0, idx1)
        y2 = torch.index_select(y, 0, idx2)
        # 计算 intra dis，类内距离
        rep_intra1 = torch.matmul(rep_x_p1, rep_x_p1.T) / temperature
        rep_intra2 = torch.matmul(rep_x_p2, rep_x_p2.T) / temperature
        # Log-sum trick for numerical stability
        # 计算内积距离后，最近的值为 1/ self.temperature，所以减去最大值后，最近为 0，远离为负值
        logits_max1, _ = torch.max(rep_intra1, dim=1, keepdim=True)
        logits_max2, _ = torch.max(rep_intra2, dim=1, keepdim=True)
        logits_intra1 = rep_intra1 - logits_max1.detach()
        logits_intra2 = rep_intra2 - logits_max2.detach()
        exp_logits1 = torch.exp(logits_intra1)
        exp_logits2 = torch.exp(logits_intra2)

        # 计算 inter dis
        rep_inter = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
        logits_max, _ = torch.max(rep_inter, dim=1, keepdim=True)
        logits_inter = rep_inter - logits_max1.detach()
        exp_inter = torch.exp(logits_inter)

        # Libo 老师讨论后, 计算 intra 相似度
        exp_logits1 = alpha * exp_logits1 / exp_inter.sum(dim=1)
        exp_logits2 = alpha * exp_logits2 / exp_inter.sum(dim=0)
        prob = exp_logits1.sum() + exp_logits2.sum()

        # Mean log-likelihood for positive
        loss_el = - (torch.log((prob))) / (len_1+len_2)

    # inter loss，类间距离
    loss = loss_natural + loss_el
    return loss

# 针对特定 label ST, 调整 conflict pair 之间 feature 的距离
# 3-5 的类内相似度大，类间相似度为 0，正交。并且不再减 1
# 仅类间相似度，取绝对值
def st_el_li3(model, x_natural, y, list_aug, alpha, temperature):
    # temperature = 0.1
    idx1 = []
    idx2 = []
    idx1.append((y == 3).nonzero().flatten())
    idx2.append((y == 5).nonzero().flatten())
    idx1 = torch.cat(idx1)
    idx2 = torch.cat(idx2)
    len_1 = len(idx1)
    len_2 = len(idx2)

    rep_x, logits_x = model(x_natural)
    loss_natural = F.cross_entropy(logits_x, y)

    if len_1 == 0 or len_2 == 0:
        loss_el = 0
        print(len_1, len_2)
    else:
        rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
        rep_x = F.normalize(rep_x.squeeze(), dim=1)
        rep_x_p1 = torch.index_select(rep_x, 0, idx1)
        rep_x_p2 = torch.index_select(rep_x, 0, idx2)
        y1 = torch.index_select(y, 0, idx1)
        y2 = torch.index_select(y, 0, idx2)
        # 计算 intra dis，类内距离 [-1, 1]
        rep_intra1 = torch.matmul(rep_x_p1, rep_x_p1.T) / temperature
        rep_intra2 = torch.matmul(rep_x_p2, rep_x_p2.T) / temperature
        # 去掉对角线的 1
        rep_intra1 = rep_intra1 - torch.eye(len(rep_x_p1)).cuda()
        rep_intra2 = rep_intra2 - torch.eye(len(rep_x_p2)).cuda()
        exp_logits1 = torch.exp(rep_intra1)
        exp_logits2 = torch.exp(rep_intra2)

        # 计算 inter sim，类间相似度
        # 取绝对值，控制在[0,1]. 使得 dis 为 0，对应 loss 最小时
        inter_sim = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
        inter_sim = torch.abs(inter_sim)
        exp_inter = torch.exp(inter_sim)

        # Libo 老师讨论后, 计算 intra 相似度
        exp_logits1 = alpha * exp_logits1 / exp_inter.sum(dim=1)
        exp_logits2 = alpha * exp_logits2 / exp_inter.sum(dim=0)
        prob = exp_logits1.sum() + exp_logits2.sum()

        # Mean log-likelihood for positive
        loss_el = - (torch.log((prob))) / (len_1+len_2)

    # inter loss，类间距离
    loss = loss_natural + loss_el
    return loss


# 针对特定 label ST, 调整 conflict pair 之间 feature 的距离
# 3-5 的类内相似度大，类间相似度为 0，正交。并且不再减 1
# 类内 & 类间相似度，都取绝对值
def st_el_li4(model, x_natural, y, list_aug, alpha, temperature):
    # temperature = 0.1
    idx1 = []
    idx2 = []
    idx1.append((y == 3).nonzero().flatten())
    idx2.append((y == 5).nonzero().flatten())
    idx1 = torch.cat(idx1)
    idx2 = torch.cat(idx2)
    len_1 = len(idx1)
    len_2 = len(idx2)

    rep_x, logits_x = model(x_natural)
    loss_natural = F.cross_entropy(logits_x, y)

    if len_1 == 0 or len_2 == 0:
        loss_el = 0
        print(len_1, len_2)
    else:
        rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
        rep_x = F.normalize(rep_x.squeeze(), dim=1)
        rep_x_p1 = torch.index_select(rep_x, 0, idx1)
        rep_x_p2 = torch.index_select(rep_x, 0, idx2)
        # 计算 intra dis，类内距离 [-1, 1]
        rep_intra1 = torch.matmul(rep_x_p1, rep_x_p1.T) / temperature
        rep_intra2 = torch.matmul(rep_x_p2, rep_x_p2.T) / temperature
        # 去掉对角线的 1
        rep_intra1 = rep_intra1 - torch.eye(len(rep_x_p1)).cuda()
        rep_intra2 = rep_intra2 - torch.eye(len(rep_x_p2)).cuda()
        # 取绝对值: [0,1]
        rep_intra1 = torch.abs(rep_intra1)
        rep_intra2 = torch.abs(rep_intra2)
        exp_logits1 = torch.exp(rep_intra1)
        exp_logits2 = torch.exp(rep_intra2)

        # 计算 inter sim，类间相似度
        # 取绝对值，控制在[0,1]. 使得 dis 为 0，对应 loss 最小时
        inter_sim = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
        inter_sim = torch.abs(inter_sim)
        exp_inter = torch.exp(inter_sim)

        # Libo 老师讨论后, 计算 intra 相似度
        exp_logits1 = alpha * exp_logits1 / exp_inter.sum(dim=1)
        exp_logits2 = alpha * exp_logits2 / exp_inter.sum(dim=0)
        prob = exp_logits1.sum() + exp_logits2.sum()

        # Mean log-likelihood for positive
        loss_el = - (torch.log((prob))) / (len_1+len_2)

    # inter loss，类间距离
    loss = loss_natural + loss_el
    return loss


# 针对特定 label ST, 调整 conflict pair 之间 feature 的距离
# 3-5 类间相似度为 0，正交。并且不再减 1
# 类内相似度不再处理
def st_el_li5(model, x_natural, y, list_aug, alpha, temperature):
    # temperature = 0.1
    idx1 = []
    idx2 = []
    idx1.append((y == 3).nonzero().flatten())
    idx2.append((y == 5).nonzero().flatten())
    idx1 = torch.cat(idx1)
    idx2 = torch.cat(idx2)
    len_1 = len(idx1)
    len_2 = len(idx2)

    rep_x, logits_x = model(x_natural)
    loss_natural = F.cross_entropy(logits_x, y)

    if len_1 == 0 or len_2 == 0:
        loss_el = 0
        print(len_1, len_2)
    else:
        rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
        rep_x = F.normalize(rep_x.squeeze(), dim=1)
        rep_x_p1 = torch.index_select(rep_x, 0, idx1)
        rep_x_p2 = torch.index_select(rep_x, 0, idx2)

        # 计算 inter sim，类间相似度
        # 取绝对值，控制在[0,1]. 使得 dis 为 0，对应 loss 最小时
        inter_sim = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
        inter_sim = torch.abs(inter_sim)
        exp_inter = torch.exp(inter_sim)

        # Libo 老师讨论后, 计算 intra 相似度
        prob = 1 / (exp_inter.sum() * alpha)

        # Mean log-likelihood for positive
        loss_el = - (torch.log((prob))) / (len_1+len_2)

    # inter loss，类间距离
    loss = loss_natural + loss_el
    return loss

# 针对特定 label ST, 调整 conflict pair 之间 feature 的距离
# 3-5 类间、类间分为两项 loss：相似度为 0，正交
def st_el_li6(model, x_natural, y, list_aug, alpha, temperature):
    # temperature = 0.1
    idx1 = []
    idx2 = []
    idx1.append((y == 3).nonzero().flatten())
    idx2.append((y == 5).nonzero().flatten())
    idx1 = torch.cat(idx1)
    idx2 = torch.cat(idx2)
    len_1 = len(idx1)
    len_2 = len(idx2)

    rep_x, logits_x = model(x_natural)
    loss_natural = F.cross_entropy(logits_x, y)

    if len_1 == 0 or len_2 == 0:
        loss_el = 0
        print(len_1, len_2)
    else:
        rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
        rep_x = F.normalize(rep_x.squeeze(), dim=1)
        rep_x_p1 = torch.index_select(rep_x, 0, idx1)
        rep_x_p2 = torch.index_select(rep_x, 0, idx2)
        # 计算 intra dis，类内距离 [-1, 1]
        rep_intra1 = torch.matmul(rep_x_p1, rep_x_p1.T) / temperature
        rep_intra2 = torch.matmul(rep_x_p2, rep_x_p2.T) / temperature
        ## 去掉对角线的 1
        # rep_intra1 = rep_intra1 - torch.eye(len(rep_x_p1)).cuda()
        # rep_intra2 = rep_intra2 - torch.eye(len(rep_x_p2)).cuda()
        # 取绝对值: [0,1]
        rep_intra1 = torch.abs(rep_intra1)
        rep_intra2 = torch.abs(rep_intra2)
        exp_logits1 = torch.exp(rep_intra1)
        exp_logits2 = torch.exp(rep_intra2)

        # 计算 inter sim，类间相似度
        # 取绝对值，控制在[0,1]. 使得 dis 为 0，对应 loss 最小时
        inter_sim = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
        inter_sim = torch.abs(inter_sim)
        exp_inter = torch.exp(inter_sim)

        inter_loss = alpha * (exp_logits1.sum() + exp_logits2.sum())
        intra_loss = 1 / exp_inter.sum()

        # Mean log-likelihood for positive
        loss_el = - (torch.log(inter_loss) + torch.log(intra_loss)) / (len_1+len_2)
        # loss_el = - torch.log(inter_loss) + exp_inter / (len_1+len_2)

    loss = loss_natural + loss_el
    return loss


# 针对特定 label ST, 调整 conflict pair 之间 feature 的距离
# 3-5 类间、类间分为两项 loss：类间相似度 abs 为 0，正交，类内 loss [0,2]
# 不使用 exp
def st_el_li7(model, x_natural, y, list_aug, alpha, temperature):
    # temperature = 0.1
    idx1 = []
    idx2 = []
    idx1.append((y == 3).nonzero().flatten())
    idx2.append((y == 5).nonzero().flatten())
    idx1 = torch.cat(idx1)
    idx2 = torch.cat(idx2)
    len_1 = len(idx1)
    len_2 = len(idx2)

    rep_x, logits_x = model(x_natural)
    loss_natural = F.cross_entropy(logits_x, y)

    if len_1 == 0 or len_2 == 0:
        loss_el = 0
        print(len_1, len_2)
    else:
        rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
        rep_x = F.normalize(rep_x.squeeze(), dim=1)
        rep_x_p1 = torch.index_select(rep_x, 0, idx1)
        rep_x_p2 = torch.index_select(rep_x, 0, idx2)
        # 计算 intra dis，类内距离 [-1, 1]
        rep_intra1 = torch.matmul(rep_x_p1, rep_x_p1.T) / temperature
        rep_intra2 = torch.matmul(rep_x_p2, rep_x_p2.T) / temperature
        # 当大家相似度为 1 时，rep_intra1 的值为 0，不相似最小为-2
        rep_intra1 = rep_intra1 - torch.ones_like(rep_intra1).cuda()
        rep_intra2 = rep_intra2 - torch.ones_like(rep_intra2).cuda()

        # 计算 inter sim，类间相似度
        # 取绝对值，控制在[0,1]. 使得 dis 为 0，对应 loss 最小时
        inter_sim = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
        inter_sim = torch.abs(inter_sim)

        # intra 添加负号，loss [0,2]
        intra_loss = -(rep_intra1.sum() + rep_intra2.sum()) * alpha
        inter_loss = inter_sim.sum()

        # Mean log-likelihood for positive
        loss_el = (intra_loss + inter_loss) / (len_1+len_2)
        # loss_el = - torch.log(inter_loss) + exp_inter / (len_1+len_2)

    loss = loss_natural + loss_el
    return loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# 使用 label smooth loss 看指标变化，在所有 data 上做
def st_ls(model, x_natural, y, smooth):
    rep_x, logits_x = model(x_natural)
    label_smooth_loss = LabelSmoothingLoss(classes=10, smoothing=smooth)
    natural_loss = label_smooth_loss(logits_x, y)
    # l = F.cross_entropy(logits_x, y)

    # inter loss，类间距离
    loss = natural_loss
    return loss

# 使用 label smooth loss 看指标变化，在所有 data 上做
# 仅对 3，5 做 label smooth
def st_ls35(model, x_natural, y, smooth):
    idx_ls, idx_other = [], []
    ls_list = [3, 5]
    other_list = [i for i in range(10)]
    for i in ls_list:
        idx_ls.append((y == i).nonzero().flatten())
    for i in ls_list:
        other_list.remove(i)
    for i in other_list:
        idx_other.append((y == i).nonzero().flatten())
    idx_ls = torch.cat(idx_ls)
    idx_other = torch.cat(idx_other)
    # ls 使用 label smooth 的 data
    # 除 ls 之外的其他 data
    len_ls = len(idx_ls)
    len_other = len(idx_other)

    _, logits_x = model(x_natural)
    logits_ls = torch.index_select(logits_x, 0, idx_ls)
    logits_other = torch.index_select(logits_x, 0, idx_other)
    y_ls = torch.index_select(y, 0, idx_ls)
    y_other = torch.index_select(y, 0, idx_other)
    label_smooth_loss = LabelSmoothingLoss(classes=10, smoothing=smooth)
    loss_ls = label_smooth_loss(logits_ls, y_ls) * len_ls
    loss_other = F.cross_entropy(logits_other, y_other) * len_other

    # inter loss，类间距离
    loss = (loss_ls + loss_other) / (len_ls + len_other)
    return loss


# 使用 label smooth loss 看指标变化，在所有 data 上做
# 仅 [2, 3, 4, 5] 都做 label smooth
def st_ls25(model, x_natural, y, smooth):
    idx_ls, idx_other = [], []
    ls_list = [2, 3, 4, 5]
    other_list = [i for i in range(10)]
    for i in ls_list:
        idx_ls.append((y == i).nonzero().flatten())
    for i in ls_list:
        other_list.remove(i)
    for i in other_list:
        idx_other.append((y == i).nonzero().flatten())
    idx_ls = torch.cat(idx_ls)
    idx_other = torch.cat(idx_other)
    # ls 使用 label smooth 的 data
    # 除 ls 之外的其他 data
    len_ls = len(idx_ls)
    len_other = len(idx_other)

    _, logits_x = model(x_natural)
    logits_ls = torch.index_select(logits_x, 0, idx_ls)
    logits_other = torch.index_select(logits_x, 0, idx_other)
    y_ls = torch.index_select(y, 0, idx_ls)
    y_other = torch.index_select(y, 0, idx_other)
    label_smooth_loss = LabelSmoothingLoss(classes=10, smoothing=smooth)
    loss_ls = label_smooth_loss(logits_ls, y_ls) * len_ls
    loss_other = F.cross_entropy(logits_other, y_other) * len_other

    # inter loss，类间距离
    loss = (loss_ls + loss_other) / (len_ls + len_other)
    return loss

def st_reweight(model, x_natural, y, weight):
    _, out = model(x_natural)
    loss = F.cross_entropy(out, y, weight=weight.cuda())
    return loss


# 针对特定 label ST, 调整 conflict pair 之间 feature 的距离
# [3，5] ST loss 调整权重，权重改为内部调整
# TRADES + ST_el_li2
def trades_el_li2(model, x_natural, y, optimizer, alpha, temperature, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta
            _, logits = model(adv)
            _, logits_x = model(x_natural)
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(logits, dim=1),
                                           F.softmax(logits_x, dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    # ST 的 3-5 el loss
    idx1 = []
    idx2 = []
    idx1.append((y == 3).nonzero().flatten())
    idx2.append((y == 5).nonzero().flatten())
    idx1 = torch.cat(idx1)
    idx2 = torch.cat(idx2)
    len_1 = len(idx1)
    len_2 = len(idx2)

    rep_x, logits_x = model(x_natural)

    if len_1 == 0 or len_2 == 0:
        loss_el = 0
        print(len_1, len_2)
    else:
        rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
        rep_x = F.normalize(rep_x.squeeze(), dim=1)
        rep_x_p1 = torch.index_select(rep_x, 0, idx1)
        rep_x_p2 = torch.index_select(rep_x, 0, idx2)
        y1 = torch.index_select(y, 0, idx1)
        y2 = torch.index_select(y, 0, idx2)
        # 计算 intra dis，类内距离
        rep_intra1 = torch.matmul(rep_x_p1, rep_x_p1.T) / temperature
        rep_intra2 = torch.matmul(rep_x_p2, rep_x_p2.T) / temperature
        # Log-sum trick for numerical stability
        # 计算内积距离后，最近的值为 1/ self.temperature，所以减去最大值后，最近为 0，远离为负值
        logits_max1, _ = torch.max(rep_intra1, dim=1, keepdim=True)
        logits_max2, _ = torch.max(rep_intra2, dim=1, keepdim=True)
        logits_intra1 = rep_intra1 - logits_max1.detach()
        logits_intra2 = rep_intra2 - logits_max2.detach()
        exp_logits1 = torch.exp(logits_intra1)
        exp_logits2 = torch.exp(logits_intra2)

        # 计算 inter dis
        rep_inter = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
        logits_max, _ = torch.max(rep_inter, dim=1, keepdim=True)
        logits_inter = rep_inter - logits_max1.detach()
        exp_inter = torch.exp(logits_inter)

        # Libo 老师讨论后, 计算 intra 相似度
        exp_logits1 = alpha * exp_logits1 / exp_inter.sum(dim=1)
        exp_logits2 = alpha * exp_logits2 / exp_inter.sum(dim=0)
        prob = exp_logits1.sum() + exp_logits2.sum()

        # Mean log-likelihood for positive
        loss_el = - (torch.log((prob))) / (len_1 + len_2)

    # calculate robust loss
    _, logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_x, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_x, dim=1))

    loss = loss_natural + loss_el + beta * loss_robust
    return loss

def at_reweight(model, x_natural, y, optimizer, weight, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_x, y, weight=weight.cuda())
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_x, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

# V1:对 natural 和 boundary loss 都做 Reweight，使用相同的 weight
# 计算 benign acc 和 benign mean 的差值，来做 reweight
def at_all_reweight_v1(model, x_natural, y, optimizer, weight, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_x, y, weight=weight.cuda())
    # Reweight boundary loss
    # weight_all = sum(weight_adv).cpu().numpy()
    weight = weight.cpu().numpy()
    loss_robust = 0
    weight_all = 0
    for m in range(10):
        idx = [i for i, x in enumerate(y) if x == m]
        a1 = logits_adv[idx]
        a2 = logits_x[idx]
        loss_robust += criterion_kl(F.log_softmax(a1, dim=1), F.softmax(a2, dim=1)) * weight[m]
        weight_all += weight[m]*len(a1)
    loss_robust = loss_robust * (1.0 / weight_all)

    # loss_robust2 = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
    #                                                 F.softmax(logits_x, dim=1))
    loss = loss_natural + beta * loss_robust

    return loss


# 对 natural 和 boundary loss 都做 Reweight，使用不同的 weight
# V2: 计算 benign acc 和 robust mean 的差值，来做 natural & boundary reweight
# V3: 计算 benign acc 和 boundary mean 的差值，来做 natural loss reweight
def at_all_reweight_v2(model, x_natural, y, optimizer, weight_natural, weight_adv, step_size=0.003, epsilon=0.031,
                      perturb_steps=10, beta=1.0, distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_x, y, weight=weight_natural.cuda())
    # Reweight boundary loss
    # weight_all = sum(weight_adv).cpu().numpy()
    weight_adv = weight_adv.cpu().numpy()
    loss_robust = 0
    weight_all = 0
    for m in range(10):
        idx = [i for i, x in enumerate(y) if x == m]
        a1 = logits_adv[idx]
        a2 = logits_x[idx]
        loss_robust += criterion_kl(F.log_softmax(a1, dim=1), F.softmax(a2, dim=1)) * weight_adv[m]
        weight_all += weight_adv[m] * len(a1)
    loss_robust = loss_robust * (1.0 / weight_all)

    loss_robust2 = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_x, dim=1))
    loss = loss_natural + beta * loss_robust

    return loss


# V1: 只对 boundary loss 做 Reweight
# 计算 benign acc 和 benign mean 的差值，来做 reweight
def at_p2_reweight_v1(model, x_natural, y, optimizer, weight, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_x, y)
    # Reweight boundary loss
    # weight_all = sum(weight_adv).cpu().numpy()
    weight = weight.cpu().numpy()
    loss_robust = 0
    weight_all = 0
    for m in range(10):
        idx = [i for i, x in enumerate(y) if x == m]
        a1 = logits_adv[idx]
        a2 = logits_x[idx]
        loss_robust += criterion_kl(F.log_softmax(a1, dim=1), F.softmax(a2, dim=1)) * weight[m]
        weight_all += weight[m]*len(a1)
    loss_robust = loss_robust * (1.0 / weight_all)

    # loss_robust2 = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
    #                                                 F.softmax(logits_x, dim=1))
    loss = loss_natural + beta * loss_robust

    return loss

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# mixup 会同自己 label 的 data 进行混合
def mixup_st(model, x_natural, y, mixalpha):
    # generate mixed inputs, two one-hot label vectors and mixing coefficient
    inputs, targets_a, targets_b, lam = mixup_data(x_natural, y, mixalpha, use_cuda=True)

    criterion = nn.CrossEntropyLoss()

    _, logits_x = model(inputs)

    loss_func = mixup_criterion(targets_a, targets_b, lam)
    loss = loss_func(criterion, logits_x)

    loss_natural = F.cross_entropy(logits_x, y)

    return loss