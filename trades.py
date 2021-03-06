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


# ???????????? label ??? adv ??? augment????????? beta ????????????
# [2,3,4,5] trade loss ????????????
def trades_loss_adp(model, x_natural, y, optimizer, list_aug, step_size=0.003, epsilon=0.031, perturb_steps=10,
                    beta=1.0,
                    distance='l_inf', beta_aug=6.0):
    list_all = [i for i in range(10)]
    list_oth = list(set(list_all) - set(list_aug))
    idx1 = []
    idx2 = []
    for i in list_oth:
        idx1.append((y == i).nonzero().flatten())
    idx1 = torch.cat(idx1)
    for i in list_aug:
        idx2.append((y == i).nonzero().flatten())
    idx2 = torch.cat(idx2)
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
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    logits_x_p1 = torch.index_select(logits_x, 0, idx1)
    logits_x_p2 = torch.index_select(logits_x, 0, idx2)
    logits_adv_p1 = torch.index_select(logits_adv, 0, idx1)
    logits_adv_p2 = torch.index_select(logits_adv, 0, idx2)
    loss_natural = F.cross_entropy(logits_x, y)
    # [0, 1, 6, 7, 8, 9] loss
    loss_robust_p1 = (1.0 / len(idx1)) * criterion_kl(F.log_softmax(logits_adv_p1, dim=1),
                                                      F.softmax(logits_x_p1, dim=1))
    # [2,3,4,5] loss
    loss_robust_p2 = (1.0 / len(idx2)) * criterion_kl(F.log_softmax(logits_adv_p2, dim=1),
                                                      F.softmax(logits_x_p2, dim=1))
    loss = loss_natural + beta * loss_robust_p1 + beta_aug * loss_robust_p2
    return loss


# ???????????? label ??? benign ??? adv ?????? reweight??? beta ????????????
# [2,3,4,5] trade loss ????????????
def trades_st_loss_adp(model, x_natural, y, optimizer, list_aug, alpha, step_size=0.003, epsilon=0.031,
                       perturb_steps=10, beta=1.0,
                       distance='l_inf', beta_aug=6.0):
    list_all = [i for i in range(10)]
    list_oth = list(set(list_all) - set(list_aug))
    idx1 = []
    idx2 = []
    for i in list_oth:
        idx1.append((y == i).nonzero().flatten())
    idx1 = torch.cat(idx1)
    for i in list_aug:
        idx2.append((y == i).nonzero().flatten())
    idx2 = torch.cat(idx2)
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
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    logits_x_p1 = torch.index_select(logits_x, 0, idx1)
    logits_x_p2 = torch.index_select(logits_x, 0, idx2)
    y1 = torch.index_select(y, 0, idx1)
    y2 = torch.index_select(y, 0, idx2)

    logits_adv_p1 = torch.index_select(logits_adv, 0, idx1)
    logits_adv_p2 = torch.index_select(logits_adv, 0, idx2)
    # loss_natural = F.cross_entropy(logits_x, y)

    # natural loss
    loss_natural_p1 = F.cross_entropy(logits_x_p1, y1)
    loss_natural_p2 = F.cross_entropy(logits_x_p2, y2)
    loss_natural = (loss_natural_p1 * len(y1) + alpha * loss_natural_p2 * len(y2)) / len(y)
    loss = F.cross_entropy(logits_x, y)

    # [0, 1, 6, 7, 8, 9] loss
    loss_robust_p1 = (1.0 / len(idx1)) * criterion_kl(F.log_softmax(logits_adv_p1, dim=1),
                                                      F.softmax(logits_x_p1, dim=1))
    # [2,3,4,5] loss
    loss_robust_p2 = (1.0 / len(idx2)) * criterion_kl(F.log_softmax(logits_adv_p2, dim=1),
                                                      F.softmax(logits_x_p2, dim=1))

    loss = loss_natural + beta * loss_robust_p1 + beta_aug * loss_robust_p2
    return loss


# ???????????? label ??? adv ??? augment????????? perturb_steps???step_size ????????????????????? aug???
# [2,3,4,5] ?????????????????? adv data
def trades_loss_aug(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                    distance='l_inf',
                    beta_aug=6.0):  # ???????????? (20???0.003)
    # train (perturb_steps=10???step_size=0.007)???test (20???0.003)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    # ????????? label ??? idx
    idx = []
    for i in [2, 3, 4, 5]:
        idx.append((y == i).nonzero().flatten())
        # idx.cuda()
    idx = torch.cat(idx)
    x_natural_aug = torch.index_select(x_natural, 0, idx)
    x_adv_aug = x_natural_aug.detach() + 0.001 * torch.randn(x_natural_aug.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            # loss_kl.backward()
            # grad = x_adv.grad

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # ?????? aug ??? adv data
        for _ in range(perturb_steps):
            x_adv_aug.requires_grad_()
            with torch.enable_grad():
                _, out_adv_aug = model(x_adv_aug)
                _, out_nat_aug = model(x_natural_aug)
                loss_kl = criterion_kl(F.log_softmax(out_adv_aug, dim=1), F.softmax(out_nat_aug, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv_aug])[0]

            # loss_kl.backward()
            # grad = x_adv_aug.grad

            x_adv_aug = x_adv_aug.detach() + step_size * torch.sign(grad.detach())
            x_adv_aug = torch.min(torch.max(x_adv_aug, x_natural_aug - epsilon), x_natural_aug + epsilon)
            x_adv_aug = torch.clamp(x_adv_aug, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_adv_aug = Variable(torch.clamp(x_adv_aug, 0.0, 1.0), requires_grad=False)
    # ???????????? label ??? data ?????????????????? adv loss

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    logits_x_aug = torch.index_select(logits_x, 0, idx)
    _, logits_adv = model(x_adv)
    _, logits_adv_aug = model(x_adv_aug)

    loss_natural = F.cross_entropy(logits_x, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_x, dim=1))
    loss_robust_aug = (1.0 / len(idx)) * criterion_kl(F.log_softmax(logits_adv_aug, dim=1),
                                                      F.softmax(logits_x_aug, dim=1))
    loss = loss_natural + beta * loss_robust + beta_aug * loss_robust_aug
    return loss


# ?????? aug ??? adv data???????????? TRADES ?????????
def adv_aug(model, x_natural, step_size=0.003, epsilon=0.031, perturb_steps=10):
    criterion_kl = nn.KLDivLoss(size_average=False)

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            _, out_adv_aug = model(x_adv)
            _, out_nat_aug = model(x_natural)
            loss_kl = criterion_kl(F.log_softmax(out_adv_aug, dim=1), F.softmax(out_nat_aug, dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]

        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


# ???????????? label ??? adv ??? augment????????? perturb_steps???step_size ????????????????????? aug????????????trades_loss_aug???????????? label ?????????????????????
# [2,3,4,5] ?????????????????? adv data
def trades_loss_augmulti(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                         distance='l_inf',
                         beta_aug=6.0):  # ???????????? (20???0.003)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # ????????? label ??? idx
    idx = []
    for i in [2, 3, 4, 5]:
        idx.append((y == i).nonzero().flatten())
        # idx.cuda()
    idx = torch.cat(idx)
    x_natural_aug = torch.index_select(x_natural, 0, idx)
    # generate adversarial example
    # ???????????? label ??? data ?????????????????? adv
    x_adv = adv_aug(model, x_natural, step_size, epsilon, perturb_steps)
    x_adv_aug = adv_aug(model, x_natural_aug, step_size, epsilon, perturb_steps)
    x_adv_aug2 = adv_aug(model, x_natural_aug, step_size, epsilon, perturb_steps)
    x_adv_aug3 = adv_aug(model, x_natural_aug, step_size, epsilon, perturb_steps)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_adv_aug = Variable(torch.clamp(x_adv_aug, 0.0, 1.0), requires_grad=False)
    x_adv_aug2 = Variable(torch.clamp(x_adv_aug2, 0.0, 1.0), requires_grad=False)
    x_adv_aug3 = Variable(torch.clamp(x_adv_aug3, 0.0, 1.0), requires_grad=False)

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    logits_x_aug = torch.index_select(logits_x, 0, idx)
    _, logits_adv = model(x_adv)
    _, logits_adv_aug = model(x_adv_aug)
    _, logits_adv_aug2 = model(x_adv_aug2)
    _, logits_adv_aug3 = model(x_adv_aug3)

    loss_natural = F.cross_entropy(logits_x, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_x, dim=1))
    loss_robust_aug = (1.0 / len(idx)) * criterion_kl(F.log_softmax(logits_adv_aug, dim=1),
                                                      F.softmax(logits_x_aug, dim=1))
    loss_robust_aug2 = (1.0 / len(idx)) * criterion_kl(F.log_softmax(logits_adv_aug2, dim=1),
                                                       F.softmax(logits_x_aug, dim=1))
    loss_robust_aug3 = (1.0 / len(idx)) * criterion_kl(F.log_softmax(logits_adv_aug3, dim=1),
                                                       F.softmax(logits_x_aug, dim=1))
    loss = loss_natural + beta * loss_robust + beta_aug * (loss_robust_aug + loss_robust_aug2 + loss_robust_aug3)
    return loss


# ???????????? label ??? adv ??? augment  +  AT loss(PGD)
# [2,3,4,5] ?????????????????? adv data???*?????? trades ????????? attack*???
# ??????????????? work
def trades_loss_aug_pgd(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                        distance='l_inf',
                        beta_aug=6.0):  # ???????????? (20???0.003)
    # train (perturb_steps=10???step_size=0.007)???test (20???0.003)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    # ????????? label ??? idx
    idx = []
    for i in [2, 3, 4, 5]:
        idx.append((y == i).nonzero().flatten())
        # idx.cuda()
    idx = torch.cat(idx)
    x_natural_aug = torch.index_select(x_natural, 0, idx)
    y_aug = torch.index_select(y, 0, idx)
    x_adv_aug = x_natural_aug.detach() + 0.001 * torch.randn(x_natural_aug.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            # loss_kl.backward()
            # grad = x_adv.grad

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # ?????? aug ??? adv data
        for _ in range(perturb_steps):
            x_adv_aug.requires_grad_()
            with torch.enable_grad():
                _, out_adv_aug = model(x_adv_aug)
                _, out_nat_aug = model(x_natural_aug)
                loss_kl = criterion_kl(F.log_softmax(out_adv_aug, dim=1), F.softmax(out_nat_aug, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv_aug])[0]

            x_adv_aug = x_adv_aug.detach() + step_size * torch.sign(grad.detach())
            x_adv_aug = torch.min(torch.max(x_adv_aug, x_natural_aug - epsilon), x_natural_aug + epsilon)
            x_adv_aug = torch.clamp(x_adv_aug, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_adv_aug = Variable(torch.clamp(x_adv_aug, 0.0, 1.0), requires_grad=False)
    # ???????????? label ??? data ?????????????????? adv loss

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    logits_x_aug = torch.index_select(logits_x, 0, idx)
    _, logits_adv = model(x_adv)
    _, logits_adv_aug = model(x_adv_aug)

    loss_natural = F.cross_entropy(logits_x, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_x, dim=1))
    loss_robust_aug = F.cross_entropy(logits_adv_aug, y_aug)
    loss = loss_natural + beta * loss_robust + beta_aug * loss_robust_aug
    return loss


# ???????????? label ??? adv ??? augment  +  AT loss(PGD)???
# [2,3,4,5] ?????????????????? adv data???*?????? pgd ????????? attack*???
def trades_loss_aug_pgdattk(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                            distance='l_inf',
                            beta_aug=6.0):  # ???????????? (20???0.003)
    # train (perturb_steps=10???step_size=0.007)???test (20???0.003)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    # ????????? label ??? idx
    idx = []
    for i in [2, 3, 4, 5]:
        idx.append((y == i).nonzero().flatten())
        # idx.cuda()
    idx = torch.cat(idx)
    x_natural_aug = torch.index_select(x_natural, 0, idx)
    y_aug = torch.index_select(y, 0, idx)
    x_adv_aug = x_natural_aug.detach() + 0.001 * torch.randn(x_natural_aug.shape).cuda().detach()
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

        # PGD attack ????????? aug adv data
        delta = pgd_linf_rand(model, x_natural_aug, y_aug, epsilon, step_size, perturb_steps, distance)
        delta = delta.detach()
        upper_limit, lower_limit = 1, 0
        x_adv_aug = torch.clamp(x_natural_aug + delta[:x_natural_aug.size(0)], min=lower_limit, max=upper_limit)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_adv_aug = Variable(torch.clamp(x_adv_aug, 0.0, 1.0), requires_grad=False)
    # ???????????? label ??? data ?????????????????? adv loss

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    # logits_x_aug = torch.index_select(logits_x, 0, idx)
    _, logits_adv = model(x_adv)
    _, logits_adv_aug = model(x_adv_aug)

    loss_natural = F.cross_entropy(logits_x, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_x, dim=1))
    loss_robust_aug = F.cross_entropy(logits_adv_aug, y_aug)
    loss = loss_natural + beta * loss_robust + beta_aug * loss_robust_aug
    return loss


# ???????????? label ??? adv ??? augment (PGD) , ?????????????????? loss
# [2,3,4,5] ?????????????????? adv data???*?????? pgd ????????? attack*???
def trades_loss_aug_pgdattk2(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                             distance='l_inf',
                             beta_aug=6.0):  # ???????????? (20???0.003)
    # train (perturb_steps=10???step_size=0.007)???test (20???0.003)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    # ????????? label ??? idx
    idx = []
    for i in [2, 3, 4, 5]:
        idx.append((y == i).nonzero().flatten())
        # idx.cuda()
    idx = torch.cat(idx)
    x_natural_aug = torch.index_select(x_natural, 0, idx)
    y_aug = torch.index_select(y, 0, idx)
    x_adv_aug = x_natural_aug.detach() + 0.001 * torch.randn(x_natural_aug.shape).cuda().detach()
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

        # PGD attack ????????? aug adv data
        delta = pgd_linf_rand(model, x_natural_aug, y_aug, epsilon, step_size, perturb_steps, distance)
        delta = delta.detach()
        upper_limit, lower_limit = 1, 0
        x_adv_aug = torch.clamp(x_natural_aug + delta[:x_natural_aug.size(0)], min=lower_limit, max=upper_limit)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_adv_aug = Variable(torch.clamp(x_adv_aug, 0.0, 1.0), requires_grad=False)
    # ???????????? label ??? data ?????????????????? adv loss

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    logits_x_aug = torch.index_select(logits_x, 0, idx)
    _, logits_adv = model(x_adv)
    _, logits_adv_aug = model(x_adv_aug)

    loss_natural = F.cross_entropy(logits_x, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_x, dim=1))
    loss_robust_aug = (1.0 / len(y_aug)) * criterion_kl(F.log_softmax(logits_adv_aug, dim=1),
                                                        F.softmax(logits_x_aug, dim=1))
    loss = loss_natural + beta * loss_robust + beta_aug * loss_robust_aug
    return loss


# ???????????? label ??? adv ??? augment  +  AT loss(PGD)
# [2,3,4,5] ?????????????????? adv data
# ?????? label ??????????????? beta


# ???????????? label ??? ST ??? AT ?????? augment????????? perturb_steps???step_size ????????????????????? aug???
# [2,3,4,5] ?????????????????? adv data???ST ??????????????? loss
def trades_loss_augSA(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                      distance='l_inf',
                      beta_aug=6.0):  # ???????????? (20???0.003)
    # train (perturb_steps=10???step_size=0.007)???test (20???0.003)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    # ????????? label ??? idx
    idx = []
    for i in [2, 3, 4, 5]:
        idx.append((y == i).nonzero().flatten())
        # idx.cuda()
    idx = torch.cat(idx)
    x_natural_aug = torch.index_select(x_natural, 0, idx)
    y_aug = torch.index_select(y, 0, idx)
    x_adv_aug = x_natural_aug.detach() + 0.001 * torch.randn(x_natural_aug.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            # loss_kl.backward()
            # grad = x_adv.grad

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # ?????? aug ??? adv data
        for _ in range(perturb_steps):
            x_adv_aug.requires_grad_()
            with torch.enable_grad():
                _, out_adv_aug = model(x_adv_aug)
                _, out_nat_aug = model(x_natural_aug)
                loss_kl = criterion_kl(F.log_softmax(out_adv_aug, dim=1), F.softmax(out_nat_aug, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv_aug])[0]

            # loss_kl.backward()
            # grad = x_adv_aug.grad

            x_adv_aug = x_adv_aug.detach() + step_size * torch.sign(grad.detach())
            x_adv_aug = torch.min(torch.max(x_adv_aug, x_natural_aug - epsilon), x_natural_aug + epsilon)
            x_adv_aug = torch.clamp(x_adv_aug, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_adv_aug = Variable(torch.clamp(x_adv_aug, 0.0, 1.0), requires_grad=False)
    # ???????????? label ??? data ?????????????????? adv loss

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    logits_x_aug = torch.index_select(logits_x, 0, idx)
    _, logits_adv = model(x_adv)
    _, logits_adv_aug = model(x_adv_aug)

    loss_natural = F.cross_entropy(logits_x, y)
    loss_natural_aug = F.cross_entropy(logits_x_aug, y_aug)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_x, dim=1))
    loss_robust_aug = (1.0 / len(idx)) * criterion_kl(F.log_softmax(logits_adv_aug, dim=1),
                                                      F.softmax(logits_x_aug, dim=1))
    loss = loss_natural + loss_natural_aug + beta * loss_robust + beta_aug * loss_robust_aug
    return loss


# ???????????? label ST
# [2,3,4,5] ST loss ????????????
def st_adp(model, x_natural, y, list_aug, alpha):
    # ????????? label ??? idx??????????????? label ??????????????????
    list_all = [i for i in range(10)]
    list_oth = list(set(list_all) - set(list_aug))
    idx1 = []
    idx2 = []
    for i in list_oth:
        idx1.append((y == i).nonzero().flatten())
    idx1 = torch.cat(idx1)
    for i in list_aug:
        idx2.append((y == i).nonzero().flatten())
    idx2 = torch.cat(idx2)

    # calculate natural loss
    _, logits_x = model(x_natural)
    logits_x_p1 = torch.index_select(logits_x, 0, idx1)
    logits_x_p2 = torch.index_select(logits_x, 0, idx2)
    y1 = torch.index_select(y, 0, idx1)
    y2 = torch.index_select(y, 0, idx2)
    # [0, 1, 6, 7, 8, 9] loss
    # ??????????????????
    loss_natural_p1 = F.cross_entropy(logits_x_p1, y1)
    # [2,3,4,5] loss
    loss_natural_p2 = F.cross_entropy(logits_x_p2, y2)
    # loss = beta * loss_natural_p1 * len(y1) / len(y) + (1 - beta) * loss_natural_p2 * len(y2) / len(y)  # ?????????
    loss = (loss_natural_p1 * len(y1) + alpha * loss_natural_p2 * len(y2)) / len(y)  # ?????????
    return loss


# ???????????? label ST, ?????? conflict pair ?????? feature ?????????
# [2,3,4,5] ST loss ????????????
def st_el(model, x_natural, y, list_aug, alpha, temperature):
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
    rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
    rep_x = F.normalize(rep_x.squeeze(), dim=1)
    rep_x_p1 = torch.index_select(rep_x, 0, idx1)
    rep_x_p2 = torch.index_select(rep_x, 0, idx2)
    y1 = torch.index_select(y, 0, idx1)
    y2 = torch.index_select(y, 0, idx2)
    # ?????? natural loss
    loss_natural = F.cross_entropy(logits_x, y)

    # ?????? intra dis???????????????
    rep_intra1 = torch.matmul(rep_x_p1, rep_x_p1.T) / temperature
    rep_intra2 = torch.matmul(rep_x_p2, rep_x_p2.T) / temperature
    # Log-sum trick for numerical stability
    # ??????????????????????????????????????? 1/ self.temperature??????????????????????????????????????? 0??????????????????
    logits_max1, _ = torch.max(rep_intra1, dim=1, keepdim=True)
    logits_max2, _ = torch.max(rep_intra2, dim=1, keepdim=True)
    logits_intra1 = rep_intra1 - logits_max1.detach()
    logits_intra2 = rep_intra2 - logits_max2.detach()
    exp_logits1 = torch.exp(logits_intra1)
    exp_logits2 = torch.exp(logits_intra2)

    # ?????? inter dis
    rep_inter = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
    logits_max, _ = torch.max(rep_inter, dim=1, keepdim=True)
    logits_inter = rep_inter - logits_max1.detach()
    exp_inter = torch.exp(logits_inter)

    # prob = (exp_logits1 + exp_logits2) / (logits_inter).sum(1, keepdim=True)
    # ?????? intra ???????????????????????? 1 ??????????????? inter ????????????????????????
    prob = (exp_logits1.sum()-len_1 + exp_logits2.sum()-len_2) / exp_inter.sum()

    # Mean log-likelihood for positive
    loss_el = - (torch.log((prob))) / (len_1+len_2)

    # libo ??????
    exp_logits1 = exp_logits1 / exp_inter.sum(dim=0)
    exp_logits2 = exp_logits2 / exp_inter.sum(dim=1)
    prob = exp_logits1.sum() + exp_logits2.sum()

    # inter loss???????????????
    loss = loss_natural + loss_el * alpha
    return loss

# st_el ????????????
def st_el_fix(model, x_natural, y, list_aug, alpha, temperature):
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
    else:
        rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
        rep_x = F.normalize(rep_x.squeeze(), dim=1)
        rep_x_p1 = torch.index_select(rep_x, 0, idx1)
        rep_x_p2 = torch.index_select(rep_x, 0, idx2)
        y1 = torch.index_select(y, 0, idx1)
        y2 = torch.index_select(y, 0, idx2)

        # ?????? intra dis???????????????
        rep_intra1 = torch.matmul(rep_x_p1, rep_x_p1.T) / temperature
        rep_intra2 = torch.matmul(rep_x_p2, rep_x_p2.T) / temperature
        # Log-sum trick for numerical stability
        # ??????????????????????????????????????? 1/ self.temperature??????????????????????????????????????? 0??????????????????
        logits_max1, _ = torch.max(rep_intra1, dim=1, keepdim=True)
        logits_max2, _ = torch.max(rep_intra2, dim=1, keepdim=True)
        logits_intra1 = rep_intra1 - logits_max1.detach()
        logits_intra2 = rep_intra2 - logits_max2.detach()
        exp_logits1 = torch.exp(logits_intra1)
        exp_logits2 = torch.exp(logits_intra2)

        # ?????? inter dis
        rep_inter = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
        logits_max, _ = torch.max(rep_inter, dim=1, keepdim=True)
        logits_inter = rep_inter - logits_max1.detach()
        exp_inter = torch.exp(logits_inter)

        # ?????? intra ???????????????????????? 1 ??????????????? 3 ?????? 5 ??????????????????????????????????????? log???????????????
        exp_logits1 = (exp_logits1 - torch.eye(len_1).cuda()) / exp_inter.sum(dim=1)
        exp_logits2 = (exp_logits2 - torch.eye(len_2).cuda()) / exp_inter.sum(dim=0)
        loss_el = - torch.log(exp_logits1.sum()/len_1) - torch.log(exp_logits2.sum()/len_2)

    # inter loss???????????????
    loss = loss_natural + loss_el * alpha
    return loss

# conflict pair ?????????????????????
def st_el2(model, x_natural, y, list_aug, alpha):
    # ????????? label ??? idx??????????????? label ??????????????????
    temperature = 0.1
    list_all = [i for i in range(10)]
    idx_el = []
    len_el = []
    for i in list_aug:
        idx_el.append((y == i).nonzero().flatten())
        len_el.append(len(idx_el))
    idx_el = torch.cat(idx_el)

    # calculate natural loss
    _, logits_x = model(x_natural)
    logits_x_el = torch.index_select(logits_x, 0, idx_el)
    # ?????? natural loss
    loss_natural = F.cross_entropy(x_natural, y)

    # ?????????????????????
    logits_x_el = torch.matmul(logits_x_el, logits_x_el.T) / temperature
    # Log-sum trick for numerical stability
    # ??????????????????????????????????????? 1/ self.temperature??????????????????????????????????????? 0??????????????????
    logits_max, _ = torch.max(logits_x_el, dim=1, keepdim=True)
    logits_x_el = logits_x_el - logits_max.detach()
    exp_logits_el = torch.exp(logits_x_el)

    # ?????? intra dis???????????????
    exp_logits_el1 = exp_logits_el[:len_el[0], :len_el[0]]
    exp_logits_el2 = exp_logits_el[len_el[1]:, len_el[1]:]
    exp_logits_intra = exp_logits_el1 + exp_logits_el2
    # ?????? inter dis
    exp_logits_inter = exp_logits_el[:len_el[0], len_el[0]:]
    exp_logits_inter1 = exp_logits_el[len_el[1]:, :len_el[0]]

    prob = exp_logits_intra / (exp_logits_inter).sum(1, keepdim=True)

    # Mean log-likelihood for positive
    loss_el = - (torch.log((prob).sum(1))).mean()
    # inter loss???????????????
    loss = loss_natural + loss_el * alpha
    return loss


# ???????????? label ST, ?????? conflict pair ?????? feature ?????????
# [2,3,4,5] ST loss ????????????
def st_el_li(model, x_natural, y, list_aug, alpha, temperature):
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
    else:
        rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
        rep_x = F.normalize(rep_x.squeeze(), dim=1)
        rep_x_p1 = torch.index_select(rep_x, 0, idx1)
        rep_x_p2 = torch.index_select(rep_x, 0, idx2)
        y1 = torch.index_select(y, 0, idx1)
        y2 = torch.index_select(y, 0, idx2)
        # ?????? intra dis???????????????
        rep_intra1 = torch.matmul(rep_x_p1, rep_x_p1.T) / temperature
        rep_intra2 = torch.matmul(rep_x_p2, rep_x_p2.T) / temperature
        # Log-sum trick for numerical stability
        # ??????????????????????????????????????? 1/ self.temperature??????????????????????????????????????? 0??????????????????
        logits_max1, _ = torch.max(rep_intra1, dim=1, keepdim=True)
        logits_max2, _ = torch.max(rep_intra2, dim=1, keepdim=True)
        logits_intra1 = rep_intra1 - logits_max1.detach()
        logits_intra2 = rep_intra2 - logits_max2.detach()
        exp_logits1 = torch.exp(logits_intra1)
        exp_logits2 = torch.exp(logits_intra2)

        # ?????? inter dis
        rep_inter = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
        logits_max, _ = torch.max(rep_inter, dim=1, keepdim=True)
        logits_inter = rep_inter - logits_max1.detach()
        exp_inter = torch.exp(logits_inter)

        # Libo ???????????????, ?????? intra ?????????
        exp_logits1 = exp_logits1 / exp_inter.sum(dim=1)
        exp_logits2 = exp_logits2 / exp_inter.sum(dim=0)
        prob = exp_logits1.sum() + exp_logits2.sum()

        # Mean log-likelihood for positive
        loss_el = - (torch.log((prob))) / (len_1+len_2)

    # inter loss???????????????
    loss = loss_natural + loss_el * alpha
    return loss

# ???????????? label ST, ????????? conflict pair ?????????????????????????????????
# logits_inter ??????????????????
def st_only_el(model, x_natural, y, list_aug, alpha, temperature):
    # ????????? label ??? idx??????????????? label ??????????????????
    # temperature = 0.1
    list_all = [i for i in range(10)]
    list_oth = list(set(list_all) - set(list_aug))
    idx1 = []
    idx2 = []
    idx1.append((y == 3).nonzero().flatten())
    idx2.append((y == 5).nonzero().flatten())
    idx1 = torch.cat(idx1)
    idx2 = torch.cat(idx2)
    len_1 = len(idx1)
    len_2 = len(idx2)

    # calculate natural loss
    rep_x, logits_x = model(x_natural)
    rep_x = F.adaptive_avg_pool2d(rep_x, (1, 1))
    rep_x = F.normalize(rep_x.squeeze(), dim=1)
    rep_x_p1 = torch.index_select(rep_x, 0, idx1)
    rep_x_p2 = torch.index_select(rep_x, 0, idx2)
    y1 = torch.index_select(y, 0, idx1)
    y2 = torch.index_select(y, 0, idx2)
    # ?????? natural loss
    loss_natural = F.cross_entropy(logits_x, y)

    # ?????? intra dis???????????????
    rep_intra1 = torch.matmul(rep_x_p1, rep_x_p1.T) / temperature
    rep_intra2 = torch.matmul(rep_x_p2, rep_x_p2.T) / temperature
    # Log-sum trick for numerical stability
    # ??????????????????????????????????????? 1/ self.temperature??????????????????????????????????????? 0??????????????????
    logits_max1, _ = torch.max(rep_intra1, dim=1, keepdim=True)
    logits_max2, _ = torch.max(rep_intra2, dim=1, keepdim=True)
    logits_intra1 = rep_intra1 - logits_max1.detach()
    logits_intra2 = rep_intra2 - logits_max2.detach()
    exp_logits1 = torch.exp(logits_intra1)
    exp_logits2 = torch.exp(logits_intra2)

    # ?????? inter dis
    rep_inter = torch.matmul(rep_x_p1, rep_x_p2.T) / temperature
    logits_max, _ = torch.max(rep_inter, dim=1, keepdim=True)
    logits_inter = rep_inter - logits_max1.detach()
    exp_inter = torch.exp(logits_inter)

    # ?????? intra ???????????????????????? 1 ??????????????? inter ????????????????????????
    # prob = (exp_logits1.sum()-len_1 + exp_logits2.sum()-len_2) / exp_inter.sum()
    prob = 1 / exp_inter.sum()

    # Mean log-likelihood for positive
    loss_el = - (torch.log((prob))) / (len_1+len_2)
    # inter loss???????????????
    loss = loss_natural + loss_el * alpha
    return loss


# ???????????? label ST, ?????? conflict pair ?????? feature ?????????
# logits ????????????????????????????????? loss ????????? 0
# logits_inter ???????????????????????????
def st_el_logits(model, x_natural, y, list_aug, alpha, temperature):
    # ????????? label ??? idx??????????????? label ??????????????????
    list_all = [i for i in range(10)]
    list_oth = list(set(list_all) - set(list_aug))
    idx1 = []
    idx2 = []
    idx1.append((y == 3).nonzero().flatten())
    idx2.append((y == 5).nonzero().flatten())
    idx1 = torch.cat(idx1)
    idx2 = torch.cat(idx2)
    len_1 = len(idx1)
    len_2 = len(idx2)

    # calculate natural loss
    rep_x, logits_x = model(x_natural)

    logits_x_1 = torch.index_select(logits_x, 0, idx1)
    logits_x_2 = torch.index_select(logits_x, 0, idx2)

    # # ?????? intra dis???????????????
    # logits_intra1 = torch.matmul(logits_x_1, logits_x_1.T) / temperature
    # logits_intra2 = torch.matmul(logits_x_2, logits_x_2.T) / temperature
    # # ??????????????????????????????????????? 1/ self.temperature??????????????????????????????????????? 0??????????????????
    # logits_max1, _ = torch.max(logits_intra1, dim=1, keepdim=True)
    # logits_max2, _ = torch.max(logits_intra2, dim=1, keepdim=True)
    # logits_intra1 = logits_intra1 - logits_max1.detach()
    # logits_intra2 = logits_intra2 - logits_max2.detach()
    # exp_logits1 = torch.exp(logits_intra1)
    # exp_logits2 = torch.exp(logits_intra2)

    if len_1 == 0 or len_2==0:
        loss_el = 0
    else:
        # ?????? inter dis
        logits_inter = torch.matmul(logits_x_1, logits_x_2.T) / temperature
        # logits_max, _ = torch.max(logits_inter, dim=1, keepdim=True)
        # logits_inter = logits_inter - logits_max1.detach()
        exp_inter = torch.exp(logits_inter)

        # ?????? intra ???????????????????????? 1 ??????????????? inter ????????????????????????
        # prob = (exp_logits1.sum()-len_1 + exp_logits2.sum()-len_2) / exp_inter.sum()
        prob = exp_inter.sum()

        # Mean log-likelihood for positive
        loss_el = (torch.log((prob))) / (len_1+len_2)
        # inter loss???????????????

    # ?????? natural loss
    loss_natural = F.cross_entropy(logits_x, y)
    loss = loss_natural + loss_el * alpha
    return loss