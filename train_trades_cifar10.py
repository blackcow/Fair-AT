from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import time
import logging
import numpy as np
from models.wideresnet import *
from models.densenet import *
from models.preactresnet import create_network
from models.resnet import *
from trades import *
from trades_my import *
from tradesfair import trades_fair_loss
from pgd import pgd_loss
from torch.utils.tensorboard import SummaryWriter
import random
from dataset.cifar10_keeplabel import CIFAR10KP, CIFAR100KP
from dataset.cifar10_rmlabel import CIFAR10RM
from dataset.imagnette import *
from dataset.cifar100_my import CIFAR100_MY
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
# model factors
parser.add_argument('--depth', type=int, default=34, metavar='N',
                    help='model depth (default: 34)')
parser.add_argument('--widen_factor', type=int, default=10, metavar='N',
                    help='model widen_factor (default: 10)')
parser.add_argument('--droprate', type=float, default=0.0, metavar='N',
                    help='model droprate (default: 0.0)')

parser.add_argument('--AT-method', type=str, default='TRADES',
                    help='AT method')
# choices=['TRADES', 'TRADES_rm', 'TRADES_loss_adp', 'TRADES_ST_adp',
#                                                'TRADES_aug', 'TRADES_augmulti', 'TRADES_aug_pgd', 'TRADES_aug_pgdattk', 'TRADES_aug_pgdattk2',
#                                                'TRADES_el', 'TRADES_el_li2','AT_reweight',
#                                                'PGD', 'ST', 'ST_adp', 'ST_el', 'ST_only_el', 'ST_el_logits',
#                                                'ST_el_li', 'ST_el_li2', 'ST_el_fix',
#                                                'ST_label_smooth', 'ST_label_smooth35', 'ST_label_smooth25',
#                                                'ST_reweight']
# parser.add_argument('--epochs', type=int, default=76, metavar='N',
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu-id', type=str, default='0,1,2', help='gpu_id')
parser.add_argument('--epsilon', default=0.031, type=float, help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet/',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=20, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--model', default='wideresnet', choices=['wideresnet', 'densenet', 'preactresnet'],
                    help='AT model name')
parser.add_argument('--fair', type=str, help='use fair_loss, choices=[v1, v2, v3, v4]')
parser.add_argument('--fairloss', type=str, help='use fair_loss, choices=[fl1, fl2, fl3, fl4]')
parser.add_argument('--T', default=0.1, type=float, help='Temperature, default=0.07')
parser.add_argument('--lamda', default=1, type=int, help='lamda of fairloss, default=10')
parser.add_argument('--fl_lamda', default=0.1, type=float, help='lamda of fairloss, default=10')

# keep label data
# parser.add_argument('--percent', default=0.1, type=float, help='Percentage of deleted data')

# training on dataset
parser.add_argument('--dataset', default='CIFAR10',
                    choices=['CIFAR10', 'CIFAR100', 'STL10', 'Imagnette', 'SVHN', 'ImageNet10'],
                    help='train model on dataset')

# ST adp
parser.add_argument('--alpha', default=0.1, type=float, help='adaptive weights of ST')
# ST tmp
parser.add_argument('--tmp', default=0.1, type=float, help='temperature of ST')
# aug
parser.add_argument('--beta_aug', default=6.0, type=float, help='regularization, i.e., 1/lambda in TRADES')
# parser.add_argument('--list_aug', default=[2, 3, 4, 5], type=float, help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--list_aug', nargs='+', type=int)

# remove label data
parser.add_argument('--rmlabel', type=int, help='Label of the remove part of training data')
parser.add_argument('--percent', default=1, type=float, help='Percentage of deleted data')

# smooth
parser.add_argument('--smooth', default=0.1, type=float, help='parameter of label smooth loss ')
# reweight
parser.add_argument('--reweight', default=0.05, type=float, help='step size of reweight')
parser.add_argument('--test_attack', action='store_true', help='Whether to attack during the test')
parser.add_argument('--discrepancy', default=0.05, type=float, help='Threshold of discrepancy')
parser.add_argument('--start_reweight', default=0, type=int, help='Threshold of discrepancy')
#mixup
parser.add_argument('--mixalpha', default=1, type=float, help='alpha of mixup')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
print(args)
# settings save model path
factors = 'e' + str(args.epsilon) + '_depth' + str(args.depth) + '_' + 'widen' + str(
    args.widen_factor) + '_' + 'drop' + str(args.droprate)

if args.fair is not None:
    model_dir = args.model_dir + args.model + '/' + args.AT_method + \
                '_fair_' + args.fair + '_fl_' + args.fairloss + '_T' + str(args.T) + '_L' + str(args.lamda)
elif args.AT_method == 'TRADES_rm':
    model_dir = args.model_dir + args.model + '/' + args.AT_method + '_' + args.dataset + '/rm_' + str(
        args.rmlabel) + '/percent_' + str(args.percent) + '/seed' + str(args.seed)
else:
    # model_dir = args.model_dir + args.model + '/' + args.AT_method
    # model_dir = args.model_dir + args.model + '/' + args.AT_method + '_' + args.dataset + '/kplabel' + '/percent_' + str(args.percent)
    # model_dir = args.model_dir + args.model + '/' + args.AT_method + '_' + args.dataset
    model_dir = args.model_dir + args.model + '/' + args.AT_method + '_' + args.dataset + '/seed' + str(args.seed)
print(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_train_STL10 = transforms.Compose([
    transforms.RandomCrop(96, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_train_Imagenet10 = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.Resize([96, 96]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.dataset == 'CIFAR10':
    if args.rmlabel:  # 如果删除特定化 label
        trainset = CIFAR10RM(root='../data', train=True, download=True, transform=transform_train, args=args)
    else:
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == 'CIFAR100':
    # trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    trainset = CIFAR100_MY(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == 'STL10':
    # trainset = torchvision.datasets.STL10(root='../data', split='test', folds=None, transform=transform_train_STL10, target_transform=None, download=True)
    trainset = torchvision.datasets.STL10(root='../data', split='train', folds=None, transform=transform_train_STL10,
                                          target_transform=None, download=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.STL10(root='../data', split='test', folds=None, transform=transform_train_STL10,
                                         target_transform=None, download=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == 'Imagnette':
    trainset = ImagenetteTrain('train')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    val_dataset = ImagenetteTrain('val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    testset = ImagenetteTrain('val')
    # testset = ImagenetteTest()
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
elif args.dataset == 'SVHN':
    # trainset = SVHNKP(root='../data', split="train", transform=transform_train, download=True, args=args)
    trainset = torchvision.datasets.SVHN(root='../data', split="train", transform=transform_train, download=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    # extraset = torchvision.datasets.SVHN(root='../data', split="extra", transform=transform_train, download=True)
    # extra_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.SVHN(root='../data', split="test", download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == 'ImageNet10':
    traindir = '../data/ilsvrc2012/train'
    valdir = '../data/ilsvrc2012/val'
    train = torchvision.datasets.ImageFolder(traindir, transform_train_Imagenet10)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val = torchvision.datasets.ImageFolder(valdir, transform_train_Imagenet10)
    test_loader = torch.utils.data.DataLoader(val, batch_size=args.test_batch_size, shuffle=False, num_workers=4)


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def eval_train(model, device, train_loader, logger):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            _, output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    logger.info('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            _, output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    logger.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# PGD Attack
def _pgd_whitebox(model, X, y, epsilon, attack, num_steps=args.num_steps, step_size=args.step_size):
    _, out = model(X)
    # N, C, H, W = rep.size()
    # rep = rep.reshape([N, -1])
    # out = out.data.max(1)[1]
    if attack == False:
        return out, out
    elif attack == True:
        X_pgd = Variable(X.data, requires_grad=True)
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(X_pgd)[1], y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        rep_pgd, out_pgd = model(X_pgd)
        # out_pgd = out_pgd.data.max(1)[1]

        # rep_pgd = rep_pgd.reshape([N, -1])
        return out, out_pgd


# 测试每个 label 的指标，给出 weight
def eval_test_perlabel(model, device, test_loader, logger, weight, weight_adv, args):
    output_all = []
    output_adv_all = []
    target_all = []
    acc_natural_label = []  # 各 label 的 acc
    acc_adv_label = [] # 各 label 的 adv acc
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            # _, output = model(data)
            output, output_adv = _pgd_whitebox(model, data, target, epsilon=args.epsilon, attack=args.test_attack)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            output = output.data.max(1)[1]
            output_adv = output_adv.data.max(1)[1]

            correct += output.eq(target.view_as(output)).sum().item()
            output_all.append(output)
            output_adv_all.append(output_adv)
            target_all.append(target)

    # 得到每个 label 的指标
    # 计算每个类别下的 err
    output_tmp = torch.stack(output_all[:-1])
    output_adv_tmp = torch.stack(output_adv_all[:-1])
    target_tmp = torch.stack(target_all[:-1])
    # 最后一行可能不满一列的长度，单独 concat
    output_all = torch.cat((output_tmp.reshape(-1), output_all[-1]), dim=0).cpu().numpy()
    output_adv_all = torch.cat((output_adv_tmp.reshape(-1), output_adv_all[-1]), dim=0).cpu().numpy()
    target_all = torch.cat((target_tmp.reshape(-1), target_all[-1]), dim=0).cpu().numpy()
    test_avg_accuracy = (output_all == target_all).sum() / target_all.size * 100
    test_adv_avg_accuracy = (output_adv_all == target_all).sum() / target_all.size * 100

    test_loss /= len(test_loader.dataset)
    logger.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), test_avg_accuracy))

    # 获取每个 label 的 out 和 target
    for m in np.unique(target_all):
        idx = [i for i, x in enumerate(target_all) if x == m]
        output_label = output_all[idx]
        output_adv_label = output_adv_all[idx]
        target_label = target_all[idx]
        acc = (output_label == target_label).sum() / target_label.size * 100
        acc_adv = (output_adv_label == target_label).sum() / target_label.size * 100
        acc_natural_label.append(acc)
        acc_adv_label.append(acc_adv)
    diff = acc_natural_label - test_avg_accuracy
    diff_adv = acc_adv_label - test_adv_avg_accuracy
    weight_step, adv_weight_step = [], []
    # 根据同 benign mean 的差值调整权重
    for i in diff:
        if i < -args.discrepancy:  # 小于均值加权重
            weight_step.append(args.reweight)
        elif i > args.discrepancy:  # 大于均值减权重
            weight_step.append(-args.reweight)
        else:
            weight_step.append(0)
    # 根据同 adv mean 的差值调整权重
    for i in diff_adv:
        if i < -args.discrepancy:  # 小于均值加权重
            adv_weight_step.append(args.reweight)
        elif i > args.discrepancy:  # 大于均值减权重
            adv_weight_step.append(-args.reweight)
        else:
            adv_weight_step.append(0)
    print(",".join(str(round(x, 3)) for x in weight_step))
    print(",".join(str(round(x, 3)) for x in adv_weight_step))
    weight = weight + torch.tensor(weight_step)  # 更新 weight
    weight_adv = weight_adv + torch.tensor(adv_weight_step)  # 更新 weight
    # weight = torch.clamp(weight, min=0.1)  # 最小权重不得小于 0
    threshold = torch.ones_like(weight) * 0.1
    weight = torch.where(weight > 0, weight, threshold)  # 最小权重不得小于 0，最小值为 threshold
    weight_adv = torch.where(weight_adv > 0, weight_adv, threshold)  # 最小权重不得小于 0，最小值为 threshold

    # 输出 weight, weight_adv
    if 'reweight' in args.AT_method:
        logger.info('Test: Benign weight of per label:')
        logger.info(",".join(str(round(x, 3)) for x in weight.cpu().numpy()))
        logger.info('Test: Adv weight of per label:')
        logger.info(",".join(str(round(x, 3)) for x in weight_adv.cpu().numpy()))
    return test_loss, test_avg_accuracy, acc_natural_label, weight, weight_adv


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# v1-v3 使用的 fair loss，input 同 label 中心近
def FairLoss1(rep, rep_center, target):
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


# 返回更新的中心点，总计数
def update(rep_center, rep_temp, rep_num, batch_num):
    # rep_center = (rep_center * rep_num + rep_temp.sum())/(rep_num+batch_num)
    rep_center = rep_num / (rep_num + batch_num) * rep_center + rep_temp.sum() / (rep_num + batch_num)
    return rep_center


def train(args, model, device, train_loader, optimizer, epoch, logger, weight, weight_adv):
    tmprep, _ = model(torch.zeros([20, 3, 32, 32]).cuda())
    _, C, H, W = tmprep.size()
    # C,H,W=512,4,4
    model.train()
    start = time.time()
    # 初始化各 label 的 rep 的中心 [10, 640, 8, 8]
    rep_benign_center = torch.zeros([10, C * H * W]).cuda()
    rep_robust_center = torch.zeros([10, C * H * W]).cuda()
    rep_center = [rep_benign_center, rep_robust_center]
    rep_num = torch.zeros([10])

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        # calculate robust loss
        if args.AT_method == 'TRADES' and args.fair is not None:
            rep_center, loss = trades_fair_loss(args=args, model=model, x_natural=data, y=target,
                                                optimizer=optimizer, rep_center=rep_center, step_size=args.step_size,
                                                epsilon=args.epsilon,
                                                perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'TRADES' or args.AT_method == 'TRADES_rm':
            loss = trades_loss(model=model, x_natural=data, y=target,
                               optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                               perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'TRADES_ST_adp':
            loss = trades_st_loss_adp(model=model, x_natural=data, y=target,
                                      optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                      perturb_steps=args.num_steps, beta=args.beta, beta_aug=args.beta_aug,
                                      list_aug=args.list_aug, alpha=args.alpha)
        elif args.AT_method == 'TRADES_loss_adp':
            loss = trades_loss_adp(model=model, x_natural=data, y=target,
                                   optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                   perturb_steps=args.num_steps, beta=args.beta, beta_aug=args.beta_aug,
                                   list_aug=args.list_aug)
        elif args.AT_method == 'TRADES_augmulti':
            loss = trades_loss_augmulti(model=model, x_natural=data, y=target,
                                        optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                        perturb_steps=args.num_steps, beta=args.beta, beta_aug=args.beta_aug)
        elif args.AT_method == 'TRADES_aug':
            loss = trades_loss_aug(model=model, x_natural=data, y=target,
                                   optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                   perturb_steps=args.num_steps, beta=args.beta, beta_aug=args.beta_aug)
        elif args.AT_method == 'TRADES_aug_pgd':
            loss = trades_loss_aug_pgd(model=model, x_natural=data, y=target,
                                       optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                       perturb_steps=args.num_steps, beta=args.beta, beta_aug=args.beta_aug)
        elif args.AT_method == 'TRADES_aug_pgdattk':
            loss = trades_loss_aug_pgdattk(model=model, x_natural=data, y=target,
                                           optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                           perturb_steps=args.num_steps, beta=args.beta, beta_aug=args.beta_aug)
        elif args.AT_method == 'TRADES_aug_pgdattk2':
            loss = trades_loss_aug_pgdattk2(model=model, x_natural=data, y=target,
                                            optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                            perturb_steps=args.num_steps, beta=args.beta, beta_aug=args.beta_aug)
        elif args.AT_method == 'TRADES_augSA':
            loss = trades_loss_augSA(model=model, x_natural=data, y=target,
                                     optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                     perturb_steps=args.num_steps, beta=args.beta, beta_aug=args.beta_aug)
        elif args.AT_method == 'TRADES_el_li2':
            loss = trades_el_li2(model=model, x_natural=data, y=target,
                                 optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                 perturb_steps=args.num_steps, beta=args.beta, alpha=args.alpha, temperature=args.tmp)
        elif args.AT_method == 'AT_reweight':  # natural loss 做 Reweight
            loss = at_reweight(model=model, x_natural=data, y=target, weight=weight,
                               optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                               perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'AT_reweightv1':  # 两项 loss 都做 Reweight
            loss = at_all_reweight_v1(model=model, x_natural=data, y=target, weight=weight,
                                     optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                     perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'AT_reweightv2':  # 两项 loss 都做 Reweight
            loss = at_all_reweight_v2(model=model, x_natural=data, y=target, weight_natural=weight, weight_adv=weight_adv,
                                     optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                     perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'AT_p2_reweightv1':  # 只对 boundary loss 做 Reweight
            loss = at_p2_reweight_v1(model=model, x_natural=data, y=target, weight=weight,
                                     optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                                     perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'PGD':
            loss = pgd_loss(model=model, X=data, y=target, optimizer=optimizer,
                            step_size=args.step_size, epsilon=args.epsilon,
                            perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'ST' and args.fair is None:
            _, out = model(data)
            loss = F.cross_entropy(out, target)
        elif args.AT_method == 'ST_adp':
            loss = st_adp(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug)
        elif args.AT_method == 'ST_el':
            loss = st_el(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug,
                         temperature=args.tmp)
        elif args.AT_method == 'ST_el_fix':
            loss = st_el_fix(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug,
                             temperature=args.tmp)
        elif args.AT_method == 'ST_only_el':
            loss = st_only_el(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug,
                              temperature=args.tmp)
        elif args.AT_method == 'ST_el_logits':
            loss = st_el_logits(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug,
                                temperature=args.tmp)
        elif args.AT_method == 'TRADES_el':
            loss = trades_loss_el(model=model, x_natural=data, y=target, optimizer=optimizer, step_size=args.step_size,
                                  epsilon=args.epsilon,
                                  perturb_steps=args.num_steps, beta=args.beta,
                                  alpha=args.alpha, list_aug=args.list_aug, temperature=args.tmp)
        elif args.AT_method == 'ST_el_li':
            loss = st_el_li(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug,
                            temperature=args.tmp)
        elif args.AT_method == 'ST_el_li2':
            loss = st_el_li2(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug,
                             temperature=args.tmp)
        elif args.AT_method == 'ST_el_li3':
            loss = st_el_li3(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug,
                             temperature=args.tmp)
        elif args.AT_method == 'ST_el_li4':
            loss = st_el_li4(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug,
                             temperature=args.tmp)
        elif args.AT_method == 'ST_el_li5':
            loss = st_el_li5(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug,
                             temperature=args.tmp)
        elif args.AT_method == 'ST_el_li6':
            loss = st_el_li6(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug,
                             temperature=args.tmp)
        elif args.AT_method == 'ST_el_li7':
            loss = st_el_li7(model=model, x_natural=data, y=target, alpha=args.alpha, list_aug=args.list_aug,
                             temperature=args.tmp)
        elif args.AT_method == 'ST_label_smooth':
            loss = st_ls(model=model, x_natural=data, y=target, smooth=args.smooth)
        elif args.AT_method == 'ST_label_smooth35':
            loss = st_ls35(model=model, x_natural=data, y=target, smooth=args.smooth)
        elif args.AT_method == 'ST_label_smooth25':
            loss = st_ls25(model=model, x_natural=data, y=target, smooth=args.smooth)
        elif args.AT_method == 'ST_reweight':
            loss = st_reweight(model=model, x_natural=data, y=target, weight=weight)
        elif args.AT_method == 'ST_mixup':
            loss = mixup_st(model=model, x_natural=data, y=target, mixalpha=args.mixalpha)
        # 不调整顺序 这里只计算了 benign 的 rep
        elif args.AT_method == 'ST' and args.fair is not None:
            rep, out = model(data)
            # 得到 input 的 rep，归一化并展开
            N, C, H, W = rep.size()
            rep = rep.reshape([N, -1])  # [N,M] [128,40960]

            # 只考虑 batch 内部
            target_tmp = target.cpu().numpy()
            # 逻辑好像不对
            for i in range(10):
                # 获取 label i 数据的索引，找到对应的 rep
                index = np.squeeze(np.argwhere(target_tmp == i))
                index1 = torch.tensor(index).cuda()
                rep_temp = torch.index_select(rep, 0, index1)

                # 更新 label i 的中心, rep_center [10, 40960]
                # fair v1：新的中心点，占 50%的权重；如果该 batch 中没有样本，则变成原来的一半
                if args.fair == 'v1':
                    rep_center[i] = (rep_center[i] + rep_temp.mean(dim=0)) / 2

                # 当前 batch 的 data 均值，作为中心点
                elif args.fair == 'v1a':
                    rep_center[i] = rep_temp.mean(dim=0)

                # fair v2：最终每个样本，占中心点的 1/n 的权重
                elif args.fair == 'v2':
                    batch_num, _ = rep_temp.size()
                    rep_center[i] = update(rep_center[i], rep_temp, rep_num[i], batch_num)  # 更新中心点
                    rep_num[i] += batch_num

                # 同 BN 一致，之前的占 90%，新的占 10%
                elif args.fair == 'v3':
                    rep_center[i] = rep_center[i] * 0.9 + rep_temp.mean(dim=0) * 0.1

            CEloss = F.cross_entropy(out, target)
            loss = CEloss + args.lamda * FairLoss1(rep, rep_center, target)
        else:
            print('Cannot find the right training pattern！')
            exit(0)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def main():
    set_random_seed(args.seed)
    # init tensorboard
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
    # init model, ResNet18() can be also used here for training
    if args.model == 'wideresnet':
        model = nn.DataParallel(
            WideResNet(depth=args.depth, widen_factor=args.widen_factor, dropRate=args.droprate)).cuda()
    elif args.model == 'densenet':
        model = nn.DataParallel(DenseNet121().cuda())
    elif args.model == 'preactresnet':  # model 小，需要降 lr
        if args.dataset == 'CIFAR100':
            model = nn.DataParallel(create_network(num_classes=100).cuda())
        elif args.dataset == 'CIFAR10' or args.dataset == 'STL10' or args.dataset == 'Imagnette' or args.dataset == 'SVHN' or args.dataset == 'ImageNet10':
            model = nn.DataParallel(create_network(num_classes=10).cuda())
        if args.dataset == 'Imagnette' or args.dataset == 'ImageNet10':  # 图片大，原有 lr 导致 loss比较大
            args.lr = 0.005
            args.weight_decay = 5e-4
            # args.weight_decay = 1e-4 # madry 版本
        else:
            args.lr = 0.01
            args.weight_decay = 5e-4

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logger = get_logger(model_dir + '/train.log')
    weight = torch.ones(10)
    weight_adv = torch.ones(10)
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        start = time.time()
        # train(args, model, device, train_loader, optimizer, epoch, logger)
        if epoch < args.start_reweight:  # 前0 轮不更新 weight
            weight = torch.ones(10)
            # weight = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.float32)
        train(args, model, device, train_loader, optimizer, epoch, logger, weight, weight_adv)
        # train(args, model, device, train_loader, optimizer, epoch)
        end = time.time()
        tm = (end - start) / 60
        print('时间(分钟):' + str(tm))
        # evaluation on natural examples
        print('================================================================')
        _, training_accuracy = eval_train(model, device, train_loader, logger)
        # _, test_accuracy = eval_test(model, device, test_loader, logger)
        _, test_accuracy, _, weight, weight_adv = eval_test_perlabel(model, device, test_loader, logger, weight,
                                                                     weight_adv, args)  # 测试每个 label 的 acc，并给出 weight
        print('================================================================')
        graph_name = factors + '_accuracy'
        writer.add_scalars(graph_name, {'training_acc': training_accuracy, 'test_accuracy': test_accuracy}, epoch)

        # save checkpoint
        if epoch % args.save_freq == 0 or epoch == 76 or epoch == 100:
            # if epoch == 76 or epoch == 100:
            # torch.save(model.state_dict(),
            #            os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            # torch.save(optimizer.state_dict(),
            #            os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))
            # 只保存模型参数
            torch.save(model.state_dict(), os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            # 合并保存
            # checkpoint = {
            #     "net": model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     "epoch": epoch
            # }
            # torch.save(checkpoint, os.path.join(model_dir, 'ckpt-epoch{}.pt'.format(epoch)))

    writer.close()


if __name__ == '__main__':
    main()
