# -*- coding: utf-8 -*-
# @Time    : 2020/6/13 5:25 下午
# @Author  : 杨树鑫
# @FileName: visualise.py
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread as gif_imread
# from catalyst import utils
from pathlib import Path
from typing import List
import PIL, json, cv2
from PIL import Image, ImageDraw
from labelme.utils import shape_to_mask

def show_examples(name: str, image: np.ndarray, mask: np.ndarray):
    plt.figure(figsize=(10, 14))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title(f"Mask: {name}")


def show(index: int, images: List[Path], masks: List[Path], transforms=None) -> None:
    image_path = images[index]
    name = image_path.name

    image = utils.imread(image_path)
#     mask = gif_imread(masks[index])
    mask = np.load(masks[index])

    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp["image"]
        mask = temp["mask"]

    show_examples(name, image, mask)

def show_random(images: List[Path], masks: List[Path], transforms=None) -> None:
    length = len(images)
    index = random.randint(0, length - 1)
    show(index, images, masks, transforms)


def show_predict(name: str, image: np.ndarray, mask: np.ndarray, pred: np.ndarray):
    plt.figure(figsize=(10, 14))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title(f"Mask: {name}")

    plt.subplot(1, 3, 3)
    plt.imshow(pred)
    plt.title(f"Pred: {name}")

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
        if isinstance(img, str):
            img_gray = PIL.Image.fromarray(img).convert('LA')
            img_gray = np.asarray(img_gray.convert('RGB'))
        else:
            if img.ndim == 2:
                img_gray = np.stack([img, img, img], axis=2)
            else:
                img_gray = img
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    return lbl_viz

def label2color():
    lbl = np.zeros((12 * 20, 3 * 20), dtype=np.uint8)
    img = np.ones((12 * 20, 3 * 20, 3), dtype=np.uint8) * 255

    HEATMAP = [(255, 0, 0), (238, 0, 0), (205, 0, 0),
               (139, 0, 0), (205, 0, 205), (238, 0, 238), (255, 0, 255),
               (0, 0, 255), (0, 0, 238), (0, 0, 205), (0, 0, 139)]
    for i in range(12):
        lbl[i*20:(i+1)*20, 25:] = i
    lbl_viz = vis_heatmap(lbl, img, alpha=1)
    lbl_viz = Image.fromarray(lbl_viz.astype(np.uint8))
    draw = ImageDraw.Draw(lbl_viz)
    for i in range(1, 12):
        draw.text((11, i * 20 + 5), str(i-1), fill='purple')

    lbl_viz.save('label2color.png')
    print("Saved!")


def vis_heatmap(lbl, image, alpha=0.5):
    # import ipdb
    # ipdb.set_trace()

    # 深红、浅红、浅蓝、深蓝
    # HEATMAP = [(255, 255, 255), (255, 181, 197), (255, 130, 171), (255, 52, 179),
    #            (255, 62, 150), (139, 34, 82), (221, 160, 221), (233, 150, 122),
    #            (250, 128, 114), (210, 105, 30), (178, 34, 34), (165, 42, 42)]

    # 肖老师2
    # HEATMAP = [(255, 255, 255), (238, 238, 0), (205, 205, 0), (139, 139, 0),
    #            (255, 215, 0), (238, 201, 0), (205, 173, 0), (139, 117, 0),
    #            (255, 193, 37), (255, 193, 37), (238, 180, 34), (205, 155, 29)]

    # 肖老师1
    # HEATMAP = [(255, 255, 255), (255, 165, 0), (244, 164, 96), (245, 222, 179),
    #            (34, 139, 34), (0, 250, 154), (0, 100, 0), (0, 206, 209),
    #            (30, 144, 255), (0, 0, 255), (123, 104, 238), (25, 25, 112)]

    # 刘老师最新标尺
    HEATMAP = [(255, 0, 0), (238, 0, 0), (205, 0, 0),
               (139, 0, 0), (205, 0, 205), (238, 0, 238), (255, 0, 255),
               (0, 0, 255), (0, 0, 238), (0, 0, 205), (0, 0, 139), (255, 255, 255)]
    labels = np.unique(lbl)
    labels = sorted(labels)
    n_labels = len(labels)
    weight = 255 // n_labels

    colormap = np.zeros(lbl.shape + (3, ))
    # colormap[:, :, 0] = lbl * weight
    # colormap[lbl == 0] = (255, 255, 255)  # background
    for i, k in enumerate(labels):
        colormap[lbl == k] = HEATMAP[k]

    vis = alpha * colormap + (1 - alpha) * image
    vis = cv2.cvtColor(vis.astype(np.float32), cv2.COLOR_RGB2BGR)
    return vis


if __name__ == "__main__":
    label2color()