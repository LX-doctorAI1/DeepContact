'''增加计算er的面积'''
import argparse
import os, json, datetime, sys, cv2
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from glob import glob
import warnings
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
from utils.utils import check_mkdir, AverageMeter

from contact import calContactDist_er_elongation
from precess import *
from config.mrcnn_config import MitochondrionConfig, MitochondrionInferenceConfig
from utils.dataset import MitochondrionDataset
from config import opts

class Contact(object):
    def __init__(self, name, n=8):
        self.name = name
        self.n = n
        self.meters = [AverageMeter(f'{i}') for i in range(n)]

    def update(self, res, n=1):
        for i, v in enumerate(res):
            self.meters[i].update(v, n)

    def get_value(self):
        return [m.avg for m in self.meters]


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


class MyRunner(SupervisedRunner):
    @torch.no_grad()
    def forward(self, model, batch, **kwargs):
        """
        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoaders.
            **kwargs: additional parameters to pass to the model
        """
        self.model = model
        output = self._process_input(batch, **kwargs)
        output = self._process_output(output)
        return output

    def _process_input_str(self, batch, **kwargs):
        output = self.model(batch[self.input_key], **kwargs)
        return output


def get_transforms(args):
    # Cell图片需要调整分辨率到10nm
    crop_size = round(1024 * 10.0 / args.resolution)
    print('crop size', crop_size)
    transforms = albu.Compose([albu.RandomCrop(crop_size, crop_size),
                               albu.Resize(1024, 1024)])
    er_transforms = albu.Compose([albu.Normalize(),
                                  albu.pytorch.ToTensorV2()])
    return transforms, er_transforms


def load_mito_model(args):
    config = MitochondrionInferenceConfig()
    mito_model = modellib.MaskRCNN(mode="inference", config=config,
                                   model_dir=args.outputdir)
    mito_model.load_weights(args.mitomodel, by_name=True)

    return mito_model


def load_er_model(args):
    ENCODER_WEIGHTS, ENCODER_NAME = 'instagram', 'resnext101_32x8d'
    if args.er_model_type == 'unet':
        er_model = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, classes=1)
    else:
        er_model = smp.FPN(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, classes=1)
    er_model.load_state_dict(torch.load(args.ermodel)["model_state_dict"])

    return er_model


def crop_pm(image, mask):
    for i in range(3):
        image[:, :, i][mask == 0] = 255
    return image


def predict_er(runner, model, er_batch):
    er_res = runner.forward(model=model, batch=er_batch)

    logits = er_res["logits"].cpu()
    er_pred = cutils.detach(logits[0].sigmoid() > args.threshold).astype(np.int32).squeeze(0)

    return er_pred


def construct_mito_pred(image, mito_pred_r, er_pred):
    mask = mito_pred_r['masks']
    mito_pred = np.zeros((mask.shape[0], mask.shape[1]))

    areas, perimeters, ratio1s, ratio2s = [], [], [], []
    for i in range(mask.shape[2]):
        tmp = np.array(mask[:, :, i]).astype('int32')
        total_area = tmp.sum()
        if total_area == 0: continue
        overlap = er_pred * tmp
        if overlap.sum() / total_area < 0.5:
            mito_pred += tmp

        # 另外要求计算的参数
        contour = cv2.Canny(tmp.astype(np.uint8) * 255, 30, 100)
        perimeter = contour.sum() / 255

        areas.append(total_area)
        perimeters.append(perimeter)
        ratio1s.append(total_area / perimeter)
        ratio2s.append(perimeter * perimeter / (12.56 * total_area))

    image[mito_pred < 1] = 255

    return image, np.mean(ratio1s), np.mean(ratio2s)


def my_visualize(image, mito_pred, er_pred, cont_image, file_name):
    vis_mito = visualize.drew_instances(image, mito_pred['masks']).astype(np.uint8)
    vis_er = label2rgb(er_pred, 2, img=image)
    vis_img = np.concatenate((vis_mito, vis_er, cont_image), axis=1)

    # cv2.imwrite(file_name, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(file_name, vis_img)


def save_res(result, result_dist, args):
    output_dir = args.outputdir
    expe_name = args.expe_name

    d = {'File_name': [], 'Mito_number': [], 'Contact_number': [], 'Mito_length': [],
         'Contact_length': [], 'Ratio_number': [], 'Ratio_length': [], 'ER_Length': [],
         'ER_Elongation': [], 'Area/Perimeter': [], 'Form_Factor': []}
    LABLE = ['Mito_number', 'Mito_length', 'Contact_length', 'Contact_number',
             'ER_Length', 'ER_Elongation', 'Area/Perimeter', 'Form_Factor']
    for name in result:
        r = result[name]
        d['File_name'].append(r.name)
        res = r.get_value()
        for label, v in zip(LABLE, res):
            d[label].append(v)
        d['Ratio_number'].append(res[3] / res[0])
        d['Ratio_length'].append(res[2] / res[1])


    df = pd.DataFrame.from_dict(d)
    df = df.sort_values(by=['File_name'])
    df.loc[len(df.index)] = ['Mean'] + df.mean().to_list()
    output_filename = os.path.join(output_dir, f'{expe_name}_result.csv')
    df.to_csv(output_filename, index=False)

    dists = {'File_name': []}
    for name in result_dist:
        dists['File_name'].append(name)
        dist = result_dist[name].get_value()
        mito_len = result[name].meters[1].avg
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
    mito_model = load_mito_model(args)
    er_model = load_er_model(args)
    # ER的模型是基于catalyst框架的，需要一个runner
    device = cutils.get_device()
    myrunner = MyRunner(device=device, input_key="image", input_target_key="mask")

    # 获取每张图要随机取样的数量
    repeat_num = getattr(args, 'repeat_num', 5)
    result, result_dist = {}, {}

    # 计算时间
    t_preprocess, t_mito, t_er, t_vis, t_cont = 0, 0, 0, 0, 0
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
                vis_image = deepcopy(mito_image)

                mito_start = time.time()
                t_preprocess += mito_start - preprocess_start

                r = mito_model.detect([mito_image], verbose=0)[0]

                preprocess_start = time.time()
                t_mito += preprocess_start - mito_start

                # 预测ER，需要先normalize图片，再转换成Tensor
                er_image = er_transforms(image=mito_image)['image'].unsqueeze(0)
                er_batch = {"image": er_image.to(device)}

                er_start = time.time()
                t_preprocess += er_start - preprocess_start

                er_pred = predict_er(myrunner, er_model, er_batch)

                vis_start = time.time()
                t_er += vis_start - er_start

                # 将Mito的结果呈现在一张图上
                mito_pred, ratio1, ratio2 = construct_mito_pred(mito_image, r, er_pred)

                cont_start = time.time()
                t_vis += cont_start - vis_start

                # 计算contact
                # [mito_num, mito_len, cont_len, cont_num, er_len, er_elong, vis, distmap]
                res = calContactDist_er_elongation(mito_pred, er_pred, vis_image, args.px)

                t_cont += time.time() - cont_start
                # print('Cal Contact time: ', time.time() - end)
                print(f'|{i+1}/{repeat_num}|{file_name}' +
                      f'| Mito:{res[0]}| Contact:{res[3]}| Mito_len:{res[1]}| Contact_len:{res[2]}' +
                      f'| ER_len:{res[4]}| ER_Elongation:{res[5]:.2f}| Area_Perimeter: {ratio1:.2f}' +
                      f'| Form_Factor: {ratio2:.2f}')

                res_con = res[:-2]
                res_con.extend([ratio1, ratio2])
                contact.update(res_con)
                all_distmap.update(res[-1])

                # 可视化
                vis_start = time.time()
                if args.visall:
                    output_name = os.path.join(output_dir, file_name + f'_{i}.png')
                    my_visualize(vis_image, r, er_pred, res[-2], output_name)
                elif i == 0:
                    output_name = os.path.join(output_dir, file_name + '.png')
                    my_visualize(vis_image, r, er_pred, res[-2], output_name)
                t_vis += time.time() - vis_start
    except Exception as e:
        print("Error: ", e)
    finally:
        save_res(result, result_dist, args)
        print("Result are saved!")
        print(f'preprocess: {t_preprocess}s, mito: {t_mito}s, er: {t_er}s, visualize: {t_vis}s, contact: {t_cont}s')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = opts.parse_opt()

    # 每个测试放在一个单独的文件夹下
    args.expe_name = os.path.basename(os.path.normpath(args.datadir))
    args.outputdir = os.path.join(args.outputdir, args.expe_name + '_mito_er_range_10_pix_vertical')

    print(args)
    os.environ['CUDA_VISIBLE DEVICES'] = args.gpu
    # 占用GPU50%的显存
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)

    main(args)
