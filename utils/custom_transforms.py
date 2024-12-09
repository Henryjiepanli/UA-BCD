import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        img_A = np.array(img_A).astype(np.float32)
        img_B = np.array(img_B).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img_A /= 255.0
        img_A -= self.mean
        img_A /= self.std

        img_B /= 255.0
        img_B -= self.mean
        img_B /= self.std

        return {'A': img_A, 'B': img_B, 'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        img_A = np.array(img_A).astype(np.float32).transpose((2, 0, 1))
        img_B = np.array(img_B).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img_A = torch.from_numpy(img_A).float()
        img_B = torch.from_numpy(img_B).float()
        mask = torch.from_numpy(mask).float()

        return {'A': img_A, 'B': img_B, 'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        if random.random() < 0.5:
            img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
            img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'A': img_A, 'B': img_B, 'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img_A = img_A.rotate(rotate_degree, Image.BILINEAR)
        img_B = img_B.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'A': img_A, 'B': img_B, 'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        if random.random() < 0.5:
            img_A = img_A.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img_B = img_B.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'A': img_A, 'B': img_B, 'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img_A.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img_A = img_A.resize((ow, oh), Image.BILINEAR)
        img_B = img_B.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img_A.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img_A = img_A.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_B = img_B.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'A': img_A, 'B': img_B, 'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']
        w, h = img_A.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img_A = img_A.resize((ow, oh), Image.BILINEAR)
        img_B = img_B.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img_A.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img_A = img_A.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_B = img_B.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'A': img_A, 'B': img_B, 'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        mask = sample['label']

        assert img_A.size == mask.size
        assert img_B.size == mask.size

        img_A = img_A.resize(self.size, Image.BILINEAR)
        img_B = img_B.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'A': img_A, 'B': img_B, 'label': mask}