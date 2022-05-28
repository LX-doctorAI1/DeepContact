# -*- coding: utf-8 -*-
import os
import cv2
import glob
import json
import datetime
import random
import argparse
import numpy as np
import os.path as osp
from pathlib import Path
import albumentations as albu
import slidingwindow as sw
from tqdm import tqdm
from pycococreatortools import pycococreatortools
from sklearn.model_selection import train_test_split
from precess import labelme_load_mask

category_info = {'id': 1, 'is_crowd': False}

transforms = {'test': albu.Resize(1024, 1024),
              'train': albu.Compose([
                  # 因为是ER图像需要缩放，因此需要Resize到1024
                  albu.Resize(1024, 1024),
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


def get_coco():
    coco_output = {
        "info": {
            "description": "Mito Dataset",
            "url": "https://github.com/aspenstarss",
            "version": "0.1.0",
            "year": 2020,
            "contributor": "aspenstars",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        },
        "licenses": [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ],
        "categories": [
            {
                'id': 1,
                'name': 'Mito',
                'supercategory': 'Mito',
            }
        ],
        "images": [],
        "annotations": []
    }

    return coco_output


def get_rid_of(mask):
    # 删除冗余的mask层
    had = []
    for m in range(mask.shape[2]):
        if np.sum(mask[:, :, m]) != 0:
            had.append(m)
    if len(had) == 0:
        had.append(0)
    mask_crop = mask[:, :, had]
    return mask_crop


def precess(args, img_files, split):
    if len(img_files) == 0:
        return

    ann_dir = osp.join(args.output_dir, 'annotations')
    img_dir = osp.join(args.output_dir, split)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    aug_num = args.aug_num
    image_id = 0
    segmentation_id = 0
    coco_output = get_coco()
    for filename in img_files:
        # read image and mask
        print('[Info] Processing {} ...'.format(filename.name))

        image = cv2.imread(str(filename))
        mask = labelme_load_mask(str(filename.with_suffix('.json')), [
                                 'Mito'], image.shape[:-1])

        if args.pm:
            pm_mask = labelme_load_mask(str(filename.with_suffix('.json')), [
                                        'Plasma Membrane'], image.shape[:-1])
            for i in range(3):
                image[:, :, i][pm_mask[:, :, 0] == 0] = 255

        # generate slide windows for crop
        windows = sw.generate(
            image, sw.DimOrder.HeightWidthChannel, args.sample_size, args.overlap)
        for i, w in enumerate(windows):
            toprow, bottomrow, leftcol, rightcol = w.x, w.x + w.h, w.y, w.y + w.w
            # print(toprow, bottomrow, leftcol, rightcol)
            
            img_crop = image[toprow: bottomrow, leftcol: rightcol, :]
            # only save non-zero slice
            mask_crop = get_rid_of(mask[toprow: bottomrow, leftcol: rightcol])

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
                        aug_filename = filename.stem + "_%03d" % i + "_%03d" % index + '.png'
                        cv2.imwrite(osp.join(img_dir, aug_filename), img_aug)

                        img_aug_info = pycococreatortools.create_image_info(
                            image_id, os.path.basename(aug_filename), img_aug.shape[:-1])
                        coco_output['images'].append(img_aug_info)

                        for k in range(mask_aug.shape[2]):
                            binary_mask = mask_aug[:, :, k].astype(np.uint8)
                            annotation_info = pycococreatortools.create_annotation_info(
                                segmentation_id, image_id, category_info, binary_mask,
                                img_aug.shape[:-1], tolerance=2)
                            if annotation_info is not None:
                                coco_output['annotations'].append(annotation_info)
                                segmentation_id = segmentation_id + 1
                        image_id = image_id + 1
                        index += 1
                        pbar.update(1)
                    else:
                        print('====>> Hard: ', hard)
                        hard += 1
                        if hard > 30:
                            break

    with open(osp.join(ann_dir, 'instances_' + split + '.json'), 'w') as f:
        json.dump(coco_output, f)


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
        description='将数据增广并分割训练和测试集,然后转换为COCO数据集的格式')
    parser.add_argument('--set', default='sem',
                        help='sem, tem, cell')
    parser.add_argument('-d', '--input_dir', required=False, default='dataset/json',
                        help='要处理的数据集文件位置')
    parser.add_argument('-o', '--output_dir', required=False, default='dataset/mito_train',
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
