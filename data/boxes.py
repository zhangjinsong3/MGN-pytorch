from data.common import list_pictures
import cv2
from PIL import Image
import numpy as np

from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader


# TODO(zjs): dataloader repeatly load and preprocess image every epoch, which make training process slower
class Boxes(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader  # PIL load image and convert to RGB mode.
        self.width = args.width
        self.height = args.height
        self.use_resize_keep_aspect_ratio = args.resize_keep_aspect_ratio
        self.use_mask = args.use_mask

        data_path = args.datadir
        if dtype == 'train':
            if args.debug_mode:
                data_path += '/train_debug'
            else:
                data_path += '/train'
        elif dtype == 'test':
            if args.debug_mode:
                data_path += '/test_debug'
            else:
                data_path += '/test'
        else:
            if args.debug_mode:
                data_path += '/query_debug'
            else:
                data_path += '/query'

        self.imgs = [path for path in list_pictures(data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

        if self.use_mask:
            self.mask = list()
            import os
            info_path = os.path.join(args.datadir, 'Info/train.json')
            assert os.path.exists(info_path), '%s do not exits!' % info_path

            import json
            js = json.load(open(info_path, 'r'))
            images_info = js['image']

            for image_path in self.imgs:
                _, image_name = os.path.split(image_path)
                image_mask = [x['mask'] for x in images_info if x['image_name']==image_name]
                assert len(image_mask)==1, '%s does not found a unique mask!' % image_path

                self.mask.append(image_mask[0])

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)

        if self.use_mask:
            if isinstance(img, Image.Image):
                img = np.asarray(img)
            x = np.array(self.mask[index][0::2])
            y = np.array(self.mask[index][1::2])
            xy = np.vstack((x, y)).transpose(1, 0)
            mask = np.zeros_like(img)
            cv2.fillPoly(mask, np.int32([xy]), (1, 1, 1))
            img = img * mask
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.uint8(img))

        if self.use_resize_keep_aspect_ratio:
            img = self.resize_keep_aspect_ratio(img, self.width, self.height,
                                                interpolation=cv2.INTER_LINEAR, color=(0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)

        return img, target, path

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        add = 0
        if file_path.split('/')[-1].split('.')[0].split('_')[1] == 'p':
            add += 10
        return int(file_path.split('/')[-1].split('.')[0].split('_')[2]) + add

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
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
