import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


# 返回 3 个，data for ST（1）+ data for CL（2）
class ST_CL_ViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, st_transform, cl_transform):
        self.st_transform = st_transform
        self.cl_transform = cl_transform
        # self.n_views = n_views

    def __call__(self, x):
        return [self.st_transform(x), self.cl_transform(x), self.cl_transform(x)]
