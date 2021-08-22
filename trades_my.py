import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
# from pgd import attack_pgd
from pgd import *


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

