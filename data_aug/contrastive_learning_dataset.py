"""
读取两次
"""
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator, ST_CL_ViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ST_CL_Dataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        cl_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor()])
        return cl_transforms

    @staticmethod
    def standard_training_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        st_transform = transforms.Compose([transforms.RandomCrop(size=size, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor()])
        return st_transform

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ST_CL_ViewGenerator(
                                                                  self.standard_training_transform(32),
                                                                  self.get_simclr_pipeline_transform(32)),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ST_CL_ViewGenerator(
                                                              self.standard_training_transform(96),
                                                              self.get_simclr_pipeline_transform(96)),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
