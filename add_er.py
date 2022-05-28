# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import numpy as np
import os.path as osp
from pathlib import Path
import albumentations as albu
import slidingwindow as sw
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from precess import labelme_load_mask_one

SUBSETS = ['train', 'test']

transforms = {'test': albu.Resize(1024, 1024),
              'train': albu.Compose([
                  albu.Resize(1024, 1024),  # 因为是ER图像需要缩放，因此需要Resize到1024
                  # albu.RandomRotate90(),  # 随机旋转
                  albu.Flip(),
                  albu.OneOf([
                      albu.GaussNoise(),  # 向输入图像添加高斯噪声
                      albu.ElasticTransform(),  # 弹性变换
                  ], p=0.5),
                  albu.ShiftScaleRotate(
                      shift_limit=0.0625, scale_limit=0.2, rotate_limit=180, p=0.7),
                  # 仿射变换
                  # albu.RandomBrightnessContrast(),  # 亮度
                  # albu.HueSaturationValue(p=0.3),  # 饱和度
              ])}


def precess(args, img_files, split):
    if len(img_files) == 0:
        return

    aug_num = args.aug_num
    output_dir = Path(args.output_dir)
    image_dir = output_dir / split / 'image'
    mask_dir = output_dir / split / 'mask'
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for filename in img_files:
        # read image and mask
        print('[Info] Processing {} ...'.format(filename.name))
        image = cv2.imread(str(filename))
        mask = labelme_load_mask_one(str(filename.with_suffix('.json')), [
                                     'ER'], image.shape[:-1])

        if args.pm:
            pm_mask = labelme_load_mask_one(
                filename.with_suffix('.json'), ['Plasma Membrane'], image.shape[:-1])
            for i in range(3):
                image[:, :, i][pm_mask[:, :, 0] == 0] = 255

        # generate slide windows for crop
        windows = sw.generate(
            image, sw.DimOrder.HeightWidthChannel, args.sample_size, args.overlap)
        for i, w in enumerate(windows):
            toprow, bottomrow, leftcol, rightcol = w.x, w.x + w.h, w.y, w.y + w.w
            # print(toprow, bottomrow, leftcol, rightcol)

            img_crop = image[toprow: bottomrow, leftcol: rightcol, :]
            mask_crop = mask[toprow: bottomrow, leftcol: rightcol]

            index, hard = 0, 0
            with tqdm(total=aug_num) as pbar:
                while index < aug_num:
                    # augument image
                    augmented = transforms[split](
                        image=img_crop, mask=mask_crop)
                    img_aug, mask_aug = augmented['image'], augmented['mask']

                    # only save iou > threshold crop
                    iou = (1024 ** 2 - (img_aug[:, :, 0] ==
                                        255).sum()) / (1024 ** 2) if args.pm else 1

                    if iou > args.threshold:
                        mask_aug = mask_aug.astype(np.int8)

                        outname = filename.stem + "_%03d" % i + "_%03d" % index
                        cv2.imwrite(
                            osp.join(image_dir, outname + '.png'), img_aug)
                        np.save(os.path.join(
                            mask_dir, outname + '.npy'), mask_aug)
                        index += 1
                        pbar.update(1)
                    else:
                        print('====>> Hard: ', hard)
                        hard += 1
                        if hard > 30:
                            break


def main(args):
    if os.path.exists(args.output_dir):
        print("Output dir exist!")
        return
    else:
        os.makedirs(args.output_dir)

    # get all image files name
    datadir = Path(args.input_dir)
    todo_files = list(datadir.glob('*.png')) + \
        list(datadir.glob('*.jpg')) + \
        list(datadir.glob('*.tif')) + \
        list(datadir.glob('*.jpeg'))

    # random split train and test set
    if len(todo_files) > 4:
        train_files, test_files = train_test_split(
            todo_files, test_size=0.2)
    else:
        train_files, test_files = todo_files, []

    # precess train and test set
    precess(args, train_files, 'train')
    precess(args, test_files, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='将数据增广并分割训练和测试集')
    parser.add_argument('--set', default='sem',
                        help='sem, tem, cell')
    parser.add_argument('-d', '--input_dir', required=False, default='dataset/json',
                        help='要处理的数据集文件位置')
    parser.add_argument('-o', '--output_dir', required=False, default='dataset/er_train/',
                        help='数据输出文件夹')
    parser.add_argument("--pm", default=False, action='store_true',
                        help="是否只保留PM区域，加此选项为只保留PM区域")
    parser.add_argument('--aug_num', type=int, default=5,
                        help="训练集每张图增广的数量")
    parser.add_argument('--threshold', type=float, default=0.15,
                        help="如果只保留PM区域，只保留区域面积比例大于阈值的图片")
    parser.add_argument('-ss', '--sample_size', type=int, default=1024,
                        help="从原图中采样的大小（用于down sample)")
    parser.add_argument('--overlap', type=float, default=0.2,
                        help="从原图中采样重叠比例")
    args = parser.parse_args()

    # cell 数据集进行缩放
    if args.set == 'cell' and args.sample_size == 1024:
        args.sample_size = 2048

    print(args)
    main(args)
