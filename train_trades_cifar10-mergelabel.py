"""
合并 3，5 然后做AT，看指标变化情况
"""
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
from trades import trades_loss
from tradesfair import trades_fair_loss
from pgd import pgd_loss
from torch.utils.tensorboard import SummaryWriter
from dataset.cifar10_mglabel import CIFAR10MG

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
                    help='AT method', choices=['TRADES', 'PGD', 'ST'])
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
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet/',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--model', default='wideresnet', choices=['wideresnet', 'densenet', 'preactresnet'],
                    help='AT model name')
parser.add_argument('--fair', type=str, help='use fair_loss, choices=[v1, v2, v3, v4]')
parser.add_argument('--fairloss', type=str, help='use fair_loss, choices=[fl1, fl2, fl3, fl4]')
parser.add_argument('--T', default=0.1, type=float, help='Temperature, default=0.07')
parser.add_argument('--lamda', default=1, type=int, help='lamda of fairloss, default=10')
parser.add_argument('--fl_lamda', default=0.1, type=float, help='lamda of fairloss, default=10')
# merge label data
parser.add_argument('--merge-label', nargs='+', type=int, help='merge label')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
print(args)
# settings save model path
factors = 'e' + str(args.epsilon) + '_depth' + str(args.depth) + '_' + 'widen' + str(args.widen_factor) + '_' + 'drop' + str(args.droprate)
if args.fair is not None:
    model_dir = args.model_dir + args.model + '/' + args.AT_method +\
                '_fair_' + args.fair + '_fl_' + args.fairloss + '_T' + str(args.T)+'_L' + str(args.lamda) + '/' + factors
else:
    model_dir = args.model_dir + args.model + '/' + args.AT_method + '/' + \
                'mglabel_seed' + str(args.seed) + '/' + 'mglabel_' + "".join(str(id) for id in args.merge_label)
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
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# remove label data (default label = 3)
trainset = CIFAR10MG(root='../data', train=True, download=True, transform=transform_train, args=args)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

testset = CIFAR10MG(root='../data', train=False, download=True, transform=transform_test, args=args)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

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

def train(args, model, device, train_loader, optimizer, epoch, logger):
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
                               optimizer=optimizer, rep_center=rep_center, step_size=args.step_size, epsilon=args.epsilon,
                               perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'TRADES':
            loss = trades_loss(model=model, x_natural=data, y=target,
                           optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                           perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'PGD':
            loss = pgd_loss(model=model, X=data, y=target, optimizer=optimizer,
                            step_size=args.step_size, epsilon=args.epsilon,
                            perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'ST' and args.fair is None:
            _, out = model(data)
            loss = F.cross_entropy(out, target)

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
                    rep_center[i] = update(rep_center[i], rep_temp, rep_num[i], batch_num) # 更新中心点
                    rep_num[i] += batch_num

                # 同 BN 一致，之前的占 90%，新的占 10%
                elif args.fair == 'v3':
                    rep_center[i] = rep_center[i] * 0.9 + rep_temp.mean(dim=0) * 0.1

            CEloss = F.cross_entropy(out, target)
            loss = CEloss + args.lamda * FairLoss1(rep, rep_center, target)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # init tensorboard
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
    # init model, ResNet18() can be also used here for training
    if args.model == 'wideresnet':
        model = nn.DataParallel(WideResNet(depth=args.depth, widen_factor=args.widen_factor, dropRate=args.droprate)).cuda()
    elif args.model == 'densenet':
        model = nn.DataParallel(DenseNet121().cuda())
    elif args.model == 'preactresnet':# model 小，需要降 lr
        model = nn.DataParallel(create_network().cuda())
        args.lr = 0.01
        args.weight_decay = 5e-4

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logger = get_logger(model_dir + '/train.log')

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        start = time.time()
        train(args, model, device, train_loader, optimizer, epoch, logger)
        # train(args, model, device, train_loader, optimizer, epoch)
        end = time.time()
        tm = (end - start)/60
        print('时间(分钟):' + str(tm))
        # evaluation on natural examples
        print('================================================================')
        _, training_accuracy = eval_train(model, device, train_loader, logger)
        _, test_accuracy = eval_test(model, device, test_loader, logger)
        print('================================================================')
        graph_name = factors + '_accuracy'
        writer.add_scalars(graph_name, {'training_acc': training_accuracy, 'test_accuracy': test_accuracy}, epoch)

        # save checkpoint
        if epoch % args.save_freq == 0 or epoch == 74 or epoch == 75 or epoch == 76:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            # torch.save(optimizer.state_dict(),
            #            os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))

    writer.close()
if __name__ == '__main__':
    main()
