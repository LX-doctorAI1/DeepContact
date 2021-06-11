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
    return [albu.Normalize(), ToTensor()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result

def load_mask(json_name, id_list, im_shape):
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