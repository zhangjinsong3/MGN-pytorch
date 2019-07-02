"""
This script do inference on MGN model trained on Boxes dataset!

example:
python scripts/inference.py  \
--height 256 \
--width 256 \
--save Boxes_MGN_adam_margin_0.6_resize_keep_aspect_ratio \
--nGPU 1   \
--num_classes 3000 \
--resume -1
"""


import torch
import os
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.distance import cdist

import model as MGN
from option import args
import utils.utility as utility
from torchvision import transforms


def preprocess(image):
    image = image.convert('RGB')
    image = resize_keep_aspect_ratio(image, 256, 256)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    return image


def fliphor(inputs):
    inv_idx = torch.arange(inputs.size(3)-1, -1, -1).long()  # N x C x H x W
    return inputs.index_select(3, inv_idx)


def resize_keep_aspect_ratio(image, width, height, interpolation=cv2.INTER_LINEAR, color=(127, 127, 127)):
    """
    keep the image aspect ratio and pad to target size
    :return:
    """

    input_as_PIL_image = isinstance(image, Image.Image)

    if input_as_PIL_image:
        image = np.asarray(image)

    h, w, _ = image.shape
    ratio = min(height / h, width / w)
    resize_h, resize_w = int(h * ratio), int(w * ratio)
    resize_image = cv2.resize(image, (resize_w, resize_h), interpolation=interpolation)
    top = round((height - resize_h) / 2 - 0.1)
    bottom = round((height - resize_h) / 2 + 0.1)
    left = round((width - resize_w) / 2 - 0.1)
    right = round((width - resize_w) / 2 + 0.1)
    pad_image = cv2.copyMakeBorder(resize_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # cv2.namedWindow('seesee', 0)
    # cv2.imshow('seesee', pad_image)
    # cv2.waitKey(0)

    assert pad_image.shape[0] == height and pad_image.shape[1] == width, \
        'shape of pad image is wrong %s x %s' % (pad_image.shape[0], pad_image.shape[1])

    return Image.fromarray(np.uint8(pad_image)) if input_as_PIL_image else pad_image


def extract_feature(model, input):
    features = torch.FloatTensor()
    ff = torch.FloatTensor(input.size(0), 2048).zero_()
    for i in range(2):
        if i==1:
            input = fliphor(input)
        device = torch.device('cpu' if args.cpu else 'cuda')
        input_img = input.to(device)
        outputs = model(input_img)
        f = outputs[0].data.cpu()
        ff = ff + f

    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))

    features = torch.cat((features, ff), 0)
    return features


def do_inference(model, image):
    image = preprocess(image)
    image = image.unsqueeze(0)
    feature = extract_feature(model, image)

    return feature.numpy()


if __name__ == '__main__':

    ckpt = utility.checkpoint(args)
    # 搭建MGN网络
    model = MGN.Model(args, ckpt)
    model.eval()

    root_dir = '/media/zjs/A638551F3854F033/Opensource_datasets/Boxes/Image'
    for line in open('/media/zjs/A638551F3854F033/Opensource_datasets/Boxes/PLTS_negative.txt', 'r'):
        image_list = [os.path.join(root_dir, x) for x in line.strip().split(' ')[:-1]]
        change_index = line.strip().split(' ')[-1]

        first_image = image_list[0]
        image = Image.open(first_image)
        first_feature = do_inference(model, image)

        for path in image_list[1:]:
            image = Image.open(path)
            feature = do_inference(model, image)

            print('distance:', cdist(first_feature, feature))

        print('---' * 50)
