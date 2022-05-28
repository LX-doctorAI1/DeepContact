# -*- coding: utf-8 -*-
'''改进contatc 的计算过程，与之前结果有所差别，暂未启用,仍然用的旧的'''
import argparse
import os, json, datetime, sys, cv2
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from glob import glob
import warnings
import traceback
warnings.filterwarnings('ignore')

import sklearn
import torch
import tensorflow as tf

from catalyst.dl import SupervisedRunner
from catalyst.dl import utils as cutils
import segmentation_models_pytorch as smp
import albumentations as albu

from mrcnn import utils
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from myutils.utils import check_mkdir, AverageMeter

from calcinter_px import calinter
from contact import calContactDist_range_10pix_min
from precess import *
from config.mrcnn_config import MitochondrionConfig, MitochondrionInferenceConfig, LDInferenceConfig
from myutils.dataset import MitochondrionDataset
from config import opts

class Contact(object):
    def __init__(self, name):
        self.name = name
        self.n_mito = AverageMeter(name)
        self.n_cont = AverageMeter(name)
        self.mito_len = AverageMeter(name)
        self.cont_len = AverageMeter(name)
        self.dist_er = AverageMeter(name)
        self.ratio1 = AverageMeter(name)
        self.ratio2 = AverageMeter(name)

    def reset(self):
        self.n_mito.reset()
        self.n_cont.reset()
        self.mito_len.reset()
        self.cont_len.reset()
        self.dist_er.reset()
        self.ratio1.reset()
        self.ratio2.reset()

    def update(self, mito, cont, mitol, conl, dist_er, r1, r2, n=1):
        self.n_mito.update(mito, n)
        self.n_cont.update(cont, n)
        self.mito_len.update(mitol, n)
        self.cont_len.update(conl, n)
        self.dist_er.update(dist_er, n)
        self.ratio1.update(r1, n)
        self.ratio2.update(r2, n)


class DistMap():
    def __init__(self, name, n):
        self.n = n
        self.name = name
        self.meters = [AverageMeter(f'dist_{i}') for i in range(n + 1)]

    def update(self, dists, n=1):
        for i, dist in enumerate(dists):
            self.meters[i].update(dist, n)

    def get_value(self):
        return [m.avg for m in self.meters]


def get_transforms(args):
    # Cell图片需要调整分辨率到10nm
    crop_size = round(1024 * 10.0 / args.resolution)
    print('crop size', crop_size)
    transforms = albu.Compose([albu.RandomCrop(crop_size, crop_size),
                               albu.Resize(1024, 1024)])
    er_transforms = albu.Compose([albu.Normalize(),
                                  albu.pytorch.ToTensorV2()])
    return transforms, er_transforms


def load_maskrcnn_model(args, type='mito'):
    if type == 'mito':
        config = MitochondrionInferenceConfig()
        model_file = args.mitomodel
    elif type == 'ld':
        config = LDInferenceConfig()
        model_file = args.ldmodel
    model = modellib.MaskRCNN(mode="inference", config=config,
                                   model_dir=args.outputdir)
    model.load_weights(model_file, by_name=True)

    return model


def crop_pm(image, mask):
    for i in range(3):
        image[:, :, i][mask == 0] = 255
    return image


def construct_pred(image, r_mito, r_ld):
    mask = r_mito['masks']
    mito_pred = np.zeros(mask.shape[:2])

    ratio1s, ratio2s = [], []
    for i in range(mask.shape[2]):
        tmp = np.array(mask[:, :, i]).astype(np.uint8)
        total_area = tmp.sum()
        if total_area == 0:
            continue
        mito_pred += tmp

        # 另外要求计算的参数
        contour = cv2.Canny(tmp.astype(np.uint8) * 255, 30, 100)
        perimeter = contour.sum() / 255

        ratio1s.append(total_area / perimeter)
        ratio2s.append(perimeter * perimeter / (12.56 * total_area))

    image[mito_pred < 1] = 255

    ld_mask = r_ld['masks']
    ld_pred = np.zeros(ld_mask.shape[:2])

    for i in range(ld_mask.shape[2]):
        tmp = np.array(ld_mask[:, :, i]).astype(np.uint8)
        ld_pred[tmp > 0] = 1

    return image, np.mean(ratio1s), np.mean(ratio2s), ld_pred.astype(np.uint8)


def my_visualize(image, mito_pred, ld_pred, cont_image, file_name):
    vis_mito = visualize.drew_instances(image, mito_pred['masks']).astype(np.uint8)
    vis_ld = label2rgb(ld_pred, 2, img=image)
    vis_img = np.concatenate((vis_mito, vis_ld, cont_image), axis=1)

    # cv2.imwrite(file_name, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(file_name, vis_img)


def save_res(result, result_dist, args):
    output_dir = args.outputdir
    expe_name = args.expe_name

    d = {'File_name': [], 'Mito_number': [], 'Contact_numer': [], 'Mito_length': [],
         'Contact_length': [], 'Ratio_number': [], 'Ratio_length': [], 'LD_Length': [],
         'Area/Perimeter': [], 'Form_Factor': []}
    for name in result:
        r = result[name]
        d['File_name'].append(r.name)
        d['Mito_number'].append(r.n_mito.avg)
        d['Contact_numer'].append(r.n_cont.avg)
        d['Mito_length'].append(r.mito_len.avg)
        d['Contact_length'].append(r.cont_len.avg)
        d['Ratio_number'].append(r.n_cont.avg / r.n_mito.avg)
        d['Ratio_length'].append(r.cont_len.avg / r.mito_len.avg)
        d['LD_Length'].append(r.dist_er.avg)
        d['Area/Perimeter'].append(r.ratio1.avg)
        d['Form_Factor'].append(r.ratio2.avg)

    df = pd.DataFrame.from_dict(d)
    df = df.sort_values(by=['File_name'])
    df.loc[len(df.index)] = ['Mean'] + df.mean().to_list()
    output_filename = os.path.join(output_dir, f'{expe_name}_result.csv')
    df.to_csv(output_filename, index=False)

    dists = {'File_name': []}
    for name in result_dist:
        dists['File_name'].append(name)
        dist = result_dist[name].get_value()
        mito_len = result[name].mito_len.avg
        for i in range(result_dist[name].n + 1):
            if f'dist_{i}' not in dists:
                dists[f'dist_{i}'] = []
            dists[f'dist_{i}'].append(dist[i] / mito_len)

    df = pd.DataFrame.from_dict(dists)
    df = df.sort_values(by=['File_name'])
    df.loc[len(df.index)] = ['Mean'] + df.mean().to_list()
    dist_filename = os.path.join(output_dir, f'{expe_name}_result_dist.csv')
    df.to_csv(dist_filename, index=False)

    print("Result Dir:", output_filename)


def main(args):
    # 可视化后图像的文件夹
    output_dir = os.path.join(args.outputdir, 'vis')
    check_mkdir(output_dir)

    # Crop图片的transform
    transforms, er_transforms = get_transforms(args)

    # 读取数据用的
    dataset_val = MitochondrionDataset()
    dataset_val.load_Mitochondrion(dataset_dir=args.datadir, subset=args.subset)
    dataset_val.prepare()

    # 加载模型
    mito_model = load_maskrcnn_model(args, 'mito')
    ld_model = load_maskrcnn_model(args, 'ld')

    # 获取每张图要随机取样的数量
    repeat_num = getattr(args, 'repeat_num', 5)
    result, result_dist = {}, {}

    # 计算时间
    t_preprocess, t_mito, t_ld, t_vis, t_cont = 0, 0, 0, 0, 0
    try:
        for image_id in tqdm(dataset_val.image_ids):

            read_start = time.time()

            file_path = dataset_val.image_info[image_id]['path']
            file_name = os.path.split(file_path)[1][:-4]
            print(f'Proprocess {file_name}...')

            # 用于计算多张图算出来结果的均值
            contact = Contact(file_name)
            all_distmap = DistMap(file_name, args.px)
            result[file_name], result_dist[file_name] = contact, all_distmap

            # 读取图片和PM的mask
            image = dataset_val.load_image(image_id)
            mask, _ = dataset_val.load_mask(image_id)

            t_preprocess += time.time() - read_start

            for i in range(repeat_num):
                preprocess_start = time.time()
                try:
                    augmented = transforms(image=image, mask=mask)
                except:
                    print("====>> image size is too small: ", image.shape)
                    augmented = albu.RandomCrop(1024, 1024)(image=image, mask=mask)

                aug_image, aug_mask = augmented['image'], augmented['mask']

                # 预测Mito，tissue的数据在预测Mito的时候先Crop PM区域
                mito_image = deepcopy(aug_image)
                if args.model == 'tissue' or args.model == 'tem':
                    mito_image = crop_pm(mito_image, aug_mask)

                mito_start = time.time()
                t_preprocess += mito_start - preprocess_start

                r_mito = mito_model.detect([mito_image], verbose=0)[0]

                ld_start = time.time()
                t_mito += ld_start - mito_start

                # 预测ER，需要先normalize图片，再转换成Tensor
                r_ld = ld_model.detect([mito_image], verbose=0)[0]

                vis_start = time.time()
                t_ld += vis_start - ld_start

                # 将Mito的结果呈现在一张图上
                mito_pred, ratio1, ratio2, ld_pred = construct_pred(mito_image, r_mito, r_ld)

                cont_start = time.time()
                t_vis += cont_start - vis_start

                n_mito, mito_len, cont_len, n_cont, cont_image, dist_er, distmap \
                    = calContactDist_range_10pix_min(mito_pred, ld_pred, aug_image, args.px)

                t_cont += time.time() - cont_start
                print(f'|{i+1}/{repeat_num}|{file_name}' +
                      f'| Mito:{n_mito}| Contact:{n_cont}| Mito_len:{mito_len}| Contact_len:{cont_len}' +
                      f'| Dist_LD:{dist_er}| Area_Perimeter: {ratio1}| Form_Factor: {ratio2}')
                contact.update(n_mito, n_cont, mito_len, cont_len, dist_er, ratio1, ratio2)
                all_distmap.update(distmap)

                # 可视化
                vis_start = time.time()
                if args.visall:
                    output_name = os.path.join(output_dir, file_name + f'_{i}.png')
                    my_visualize(aug_image, r_mito, ld_pred, cont_image, output_name)
                elif i == 0:
                    output_name = os.path.join(output_dir, file_name + '.png')
                    my_visualize(aug_image, r_mito, ld_pred, cont_image, output_name)
                t_vis += time.time() - vis_start
    except Exception as e:
        print("Error: ", e)
        traceback.print_exc()
    finally:
        save_res(result, result_dist, args)
        print("Result are saved!")
        print(f'preprocess: {t_preprocess}s, mito: {t_mito}s, er: {t_ld}s, visualize: {t_vis}s, contact: {t_cont}s')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = opts.parse_opt()

    # 每个测试放在一个单独的文件夹下
    args.expe_name = os.path.basename(os.path.normpath(args.datadir))
    args.outputdir = os.path.join(args.outputdir, args.expe_name + '_mito_ld_range_10_pix_vertical')

    print(args)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # 占用GPU50%的显存
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)

    main(args)
