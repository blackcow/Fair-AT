from PIL import Image
import os
import os.path
import numpy as np
import pickle


import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        # self.args = args

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100_MY(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        # 'key': 'fine_label_names',
        'key': 'coarse_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


# '''
# ?????????????????????pytorch
#     meta
#     test
#     train
# '''
#
# import os
# import cv2
# import pickle
# import time
# import numpy as np
# import matplotlib.pyplot as plt
#
# import torchvision
# from torch.autograd import Variable
# import torch.utils.data as Data
# from torchvision import transforms
#
#
# def load_CIFAR_100(root, train=True, fine_label=True):
#     """
#     root,?????????
#     train  ?????????????????????True??????????????????False
#     fine_label  ???????????????100?????????True????????????20?????????False
#      """
#     if train:
#         filename = root + 'train'
#     else:
#         filename = root + 'test'
#
#     with open(filename, 'rb')as f:
#         datadict = pickle.load(f, encoding='bytes')
#
#         X = datadict[b'data']
#
#         if train:
#             # [50000, 32, 32, 3]
#             X = X.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
#         else:
#             # [10000, 32, 32, 3]
#             X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
#
#         # fine_labels???????????????100?????????
#         # coarse_labels???????????????20??????????????????????????????????????????5???fine_labels
#         # ???trees??????????????????maple, oak, palm, pine, willow???5???????????????
#         # ????????????fine_labels
#         # Y = datadict[b'coarse_labels']+datadict[b'fine_labels']
#         if fine_label:
#             Y = datadict[b'fine_labels']
#         else:
#             Y = datadict[b'coarse_labels']
#
#         Y = np.array(Y)
#         return X, Y
#
#
# class DealDataset(Data.Dataset):
#     """
#         ??????????????????????????????
#     """
#
#     def __init__(self, root, train=True, fine_label=True, transform=None):
#         # ???????????????????????????torch.load(),????????????????????????torch.Tensor??????
#         self.x, self.y = load_CIFAR_100(root, train=train, fine_label=fine_label)
#
#         self.transform = transform
#         self.train = train
#
#     def __getitem__(self, index):
#         img, target = self.x[index], int(self.y[index])
#
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, target
#
#     def __len__(self):
#         return len(self.x)
#
#
# root = r'E:\cifar-100-python' + '/'
# batch_size = 20
#
# # ?????????????????????????????????????????????Dataset????????????????????????????????????????????????DataLoader??????????????????
# trainDataset = DealDataset(root, train=True, fine_label=True, transform=transforms.ToTensor())
# testDataset = DealDataset(root, train=False, fine_label=True, transform=transforms.ToTensor())
#
# # ????????????????????????????????????
# train_loader = Data.DataLoader(
#     dataset=trainDataset,
#     batch_size=batch_size,  # ?????????????????????????????????????????????????????????batch_size?????????
#     shuffle=False,
# )
#
# test_loader = Data.DataLoader(
#     dataset=testDataset,
#     batch_size=batch_size,
#     shuffle=False,
# )
#
# if __name__ == '__main__':
#     # ??????trainDataset??????:train_labels, train_set?????????;  ??????????????????ndarray
#     print(f'trainDataset.y.shape:{trainDataset.y.shape}\n')
#     print(f'trainDataset.y.shape:{trainDataset.x.shape}\n')
#
#     # ??????train_loader??????:batch_size???dataset?????????????????????????????????int???DealDataset
#     # dataset????????????train_labels, train_set?????????;  ??????????????????ndarray
#     print(f'train_loader.batch_size: {train_loader.batch_size}\n')
#     print(f'train_loader.dataset.y.shape: {train_loader.dataset.y.shape}\n')
#     print(f'train_loader.dataset.x.shape: {train_loader.dataset.x.shape}\n')
#
#     # # --?????????1,??????OpenCV----------------------------------------------
#     images, lables = next(iter(train_loader))
#     img = torchvision.utils.make_grid(images, nrow=10)
#     img = img.numpy().transpose(1, 2, 0)
#     # OpenCV?????????BGR?????????img???RGB?????????????????????img[:,:,::-1]
#     cv2.imshow('img', img[:, :, ::-1])
#     cv2.waitKey(0)

# ???????????????
# """ train and test dataset
# author baiyu
# """
# import os
# import sys
# import pickle
#
# from skimage import io
# import matplotlib.pyplot as plt
# import numpy
# import torch
# from torch.utils.data import Dataset
#
# class CIFAR100Train(Dataset):
#     """cifar100 test dataset, derived from
#     torch.utils.data.DataSet
#     """
#
#     def __init__(self, path, transform=None):
#         #if transform is given, we transoform data using
#         with open(os.path.join(path, 'train'), 'rb') as cifar100:
#             self.data = pickle.load(cifar100, encoding='bytes')
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data['fine_labels'.encode()])
#
#     def __getitem__(self, index):
#         label = self.data['fine_labels'.encode()][index]
#         r = self.data['data'.encode()][index, :1024].reshape(32, 32)
#         g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
#         b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
#         image = numpy.dstack((r, g, b))
#
#         if self.transform:
#             image = self.transform(image)
#         return label, image
#
# class CIFAR100Test(Dataset):
#     """cifar100 test dataset, derived from
#     torch.utils.data.DataSet
#     """
#
#     def __init__(self, path, transform=None):
#         with open(os.path.join(path, 'test'), 'rb') as cifar100:
#             self.data = pickle.load(cifar100, encoding='bytes')
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data['data'.encode()])
#
#     def __getitem__(self, index):
#         label = self.data['fine_labels'.encode()][index]
#         r = self.data['data'.encode()][index, :1024].reshape(32, 32)
#         g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
#         b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
#         image = numpy.dstack((r, g, b))
#
#         if self.transform:
#             image = self.transform(image)
#         return label, image
