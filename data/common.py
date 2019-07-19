#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import torch
import random
from PIL import Image
import matplotlib.pyplot as plt


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

    return sorted([os.path.join(root, f)
                   for root, _, files in os.walk(directory) for f in files
                   if re.match(r'([\w]+\.(?:' + ext + '))', f)])


def visual_distribution(data):
    import matplotlib
    matplotlib.use('Qt5Agg')

    def normfun(x, mu, sigma):
        pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
        return pdf

    data = np.asarray(data)
    mean = data.mean()
    std = data.std()
    print('mean: %2.2f, std: %2.2f' % (mean, std))

    # 拟合的正态分布曲线
    x = np.arange(0, 1, 0.01)
    y = normfun(x, mean, std)
    plt.plot(x, y)

    # 数据分布图
    plt.hist(data, bins=20, density=True)  # bins:柱数,
    plt.title('distance distribution')
    plt.xlabel('distance')
    plt.ylabel('probability')
    plt.show()

    return mean, std


class Random2DTranslation(object):
    """Randomly translates the input image with a probability.

    Specifically, given a predefined shape (height, width), the input is first
    resized with a factor of 1.25, leading to (height*1.25, width*1.25), then
    a random crop is performed. Such operation is done with a probability.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


class ColorAugmentation(object):
    """Randomly alters the intensities of RGB channels.

    Reference:
        Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural
        Networks. NIPS 2012.

    Args:
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor([[0.4009, 0.7192, -0.5675], [-0.8140, -0.0045, -0.5808], [0.4203, -0.6948, -0.5836],])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor
