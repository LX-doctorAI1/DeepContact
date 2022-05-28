# -*- coding: utf-8 -*-
# @Time    : 2020/6/13 5:32 下午
# @Author  : 杨树鑫
# @FileName: precess.py
import os

import PIL
import albumentations as albu
from albumentations.pytorch import ToTensor
from labelme.utils import shape_to_mask
import numpy as np
import json

def pre_transforms(image_size=1024):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
        albu.RandomRotate90(),
        albu.Cutout(),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3),
        albu.HueSaturationValue(p=0.3)
    ]

    return result


def resize_transforms(image_size=1024):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
        albu.SmallestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
        albu.LongestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    # Converts the image to a square of size image_size x image_size
    result = [
        albu.OneOf([
            random_crop,
            rescale,
            random_crop_big
        ], p=1)
    ]

    return result


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), albu.pytorch.ToTensorV2()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result

# def load_mask(json_name, id_list, im_shape):
#     with open(json_name, 'r') as f:
#         json_data = json.load(f)
#     mask = np.zeros(im_shape)
#     for shape in json_data['shapes']:
#         if shape['label'] not in id_list: continue
#         try:
#             enc = shape_to_mask(im_shape, points=shape['points'])
#         except:
#             continue
#         mask[enc] = 1
#     return mask

# ## My utils
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

from labelme.utils import shape_to_mask
import json


def labelme_load_mask(json_name, label_list, im_shape):
    """
    将每个标注存一张图，用于实例分割
    """
    mask = []
    with open(json_name) as f:
        json_data = json.load(f)
        for shape in json_data['shapes']:
            if(shape['label'] not in label_list): continue
            rle = np.zeros(im_shape)
            try:
                enc = shape_to_mask(im_shape, points=shape['points'])
            except:
                continue
            rle[enc] = 1
            mask.append(rle)

        if len(mask) != 0:
            mask = np.stack(mask, axis=-1)
        else:
            if label_list[0] == "Plasma Membrane":
                mask = np.ones((im_shape[0], im_shape[0], 1), dtype=np.uint8)
            else:
                mask = np.zeros((im_shape[0], im_shape[0], 1), dtype=np.uint8)
    return mask


def labelme_load_mask_one(json_name, id_list, im_shape):
    """
    将所有标注存在一张图上，用于语义分割
    """
    with open(json_name, 'r') as f:
        json_data = json.load(f)
    mask = np.zeros(im_shape)
    for shape in json_data['shapes']:
        if shape['label'] not in id_list: continue
        try:
            enc = shape_to_mask(im_shape, points=shape['points'])
        except:
            continue
        mask[enc] = 1
    return mask


def label_colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def _validate_colormap(colormap, n_labels):
    if colormap is None:
        colormap = label_colormap(n_labels)
    else:
        assert colormap.shape == (colormap.shape[0], 3), \
            'colormap must be sequence of RGB values'
        assert 0 <= colormap.min() and colormap.max() <= 1, \
            'colormap must ranges 0 to 1'
    return colormap


def label2rgb(lbl, n_labels=None, img=None, alpha=0.5, colormap=None):
    if n_labels is None:
        n_labels = len(np.unique(lbl))

    colormap = _validate_colormap(colormap, n_labels)
    colormap = (colormap * 255).astype(np.uint8)

    lbl_viz = colormap[lbl]
    lbl_viz[lbl == 0] = (255, 255, 255)  # background
    if img is not None:
        img_gray = PIL.Image.fromarray(img).convert('LA')
        img_gray = np.asarray(img_gray.convert('RGB'))
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    return lbl_viz