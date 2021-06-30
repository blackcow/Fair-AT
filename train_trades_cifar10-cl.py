"""
TRADES 里，加入 Contrastive Loss
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
from models.wideresnet_cl import *
from models.densenet import *
from models.preactresnet_cl import create_network
from models.resnet import *
from trades import trades_loss
# from tradesfair import trades_fair_loss
from tradescl import trades_cl_loss
from pgd import pgd_loss
from torch.utils.tensorboard import SummaryWriter
from data_aug.contrastive_learning_dataset import ST_CL_Dataset
from tqdm import tqdm

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
# contrastive learning
parser.add_argument('--cl', type=str, help='use cl_loss, choices=[v1, v2, v3, v4]')
parser.add_argument('--dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
args = parser.parse_args()

# --cl True --model preactresnet --gpu-id 2,3 --AT-method ST

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
print(args)
# settings save model path
factors = 'e' + str(args.epsilon) + '_depth' + str(args.depth) + '_' + 'widen' + str(
    args.widen_factor) + '_' + 'drop' + str(args.droprate)
model_dir = args.model_dir + args.model + '/' + args.AT_method + '_cl'

print(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# setup data loader
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

# Load contrastive learning data (x,x+)
dataset_st_cl = ST_CL_Dataset('../data')
train_dataset_st_cl = dataset_st_cl.get_dataset(args.dataset_name, args.n_views)
train_loader_st_cl = torch.utils.data.DataLoader(
    train_dataset_st_cl, batch_size=args.batch_size, shuffle=True,
    num_workers=12, pin_memory=True, drop_last=True)


def info_nce_loss(features):
    labels = torch.cat([torch.arange(args.batch_size) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix_new = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives [512, 1]
    positives = similarity_matrix_new[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives [512, 510]
    negatives = similarity_matrix_new[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)  # [512, 511]
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / args.temperature
    return logits, labels


def eval_train(model, device, train_loader, logger):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data_st_cl, target in train_loader:
            st_data = data_st_cl[0]
            st_data, target = st_data.cuda(), target.cuda()
            _, output = model(st_data)
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


# 返回更新的中心点，总计数
def update(rep_center, rep_temp, rep_num, batch_num):
    # rep_center = (rep_center * rep_num + rep_temp.sum())/(rep_num+batch_num)
    rep_center = rep_num / (rep_num + batch_num) * rep_center + rep_temp.sum() / (rep_num + batch_num)
    return rep_center


def train(args, model, device, train_loader, optimizer, epoch, logger):
    # C,H,W=512,4,4
    model.train()
    start = time.time()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for batch_idx, (data_st_cl, target) in enumerate(train_loader):
        st_data, data_cl, data_cl2 = data_st_cl[0], data_st_cl[1], data_st_cl[2]
        st_data, target = st_data.cuda(), target.cuda()
        data_cl = [data_cl, data_cl2]
        images = torch.cat(data_cl, dim=0)

        optimizer.zero_grad()
        # calculate robust loss
        if args.AT_method == 'TRADES' and args.cl is not None:
            rep_center, loss = trades_cl_loss(args=args, model=model, x_natural=data, y=target,
                                              optimizer=optimizer, rep_center=rep_center, step_size=args.step_size,
                                              epsilon=args.epsilon,
                                              perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'TRADES':
            loss = trades_loss(model=model, x_natural=data, y=target,
                               optimizer=optimizer, step_size=args.step_size, epsilon=args.epsilon,
                               perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'PGD':
            loss = pgd_loss(model=model, X=data, y=target, optimizer=optimizer,
                            step_size=args.step_size, epsilon=args.epsilon,
                            perturb_steps=args.num_steps, beta=args.beta)
        elif args.AT_method == 'ST' and args.cl is not None:
            _, out = model(st_data)
            CEloss = F.cross_entropy(out, target)
            # with autocast(enabled=self.args.fp16_precision):
            # features, _ = model(images)  # 这里 feature 的设计需要考虑下  [batch, 512, 1, 1]
            # B, _, _, _ = features.size()
            # features = features.reshape(B, -1)
            features, _ = model(images)  # 这里 feature 是加了 cl 的 mlp 的  [batch, out_dim]
            logits, labels = info_nce_loss(features)
            loss = CEloss + criterion(logits, labels)  # labels 都是 0

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(st_data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # init tensorboard
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
    # init model, ResNet18() can be also used here for training
    if args.model == 'wideresnet':
        model = nn.DataParallel(
            WideResNet(depth=args.depth, widen_factor=args.widen_factor, dropRate=args.droprate)).cuda()
    elif args.model == 'densenet':
        model = nn.DataParallel(DenseNet121().cuda())
    elif args.model == 'preactresnet':  # model 小，需要降 lr
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
        train(args, model, device, train_loader_st_cl, optimizer, epoch, logger)
        # train(args, model, device, train_loader, optimizer, epoch)
        end = time.time()
        tm = (end - start) / 60
        print('时间(分钟):' + str(tm))
        # evaluation on natural examples
        print('================================================================')
        _, training_accuracy = eval_train(model, device, train_loader_st_cl, logger)
        _, test_accuracy = eval_test(model, device, test_loader, logger)
        print('================================================================')
        graph_name = factors + '_accuracy'
        writer.add_scalars(graph_name, {'training_acc': training_accuracy, 'test_accuracy': test_accuracy}, epoch)

        # save checkpoint
        # if epoch % args.save_freq == 0 and epoch > 50 or epoch == 74 or epoch == 75 or epoch == 76:
        if epoch % args.save_freq == 0 or epoch == 74 or epoch == 75 or epoch == 76:
            # torch.save(model.state_dict(),
            #            os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            # torch.save(optimizer.state_dict(),
            #            os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))
            # 合并保存
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, os.path.join(model_dir, 'ckpt-epoch{}.pt'.format(epoch)))

    writer.close()


if __name__ == '__main__':
    main()
