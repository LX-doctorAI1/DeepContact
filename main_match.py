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

from contact import calContact, calContactDist
from precess import *
from config.mrcnn_config import MitochondrionConfig, MitochondrionInferenceConfig
from utils.dataset import ManualDataset
from config import opts
from organ_match import matching


class AvgCount(object):
    def __init__(self, name, n=1):
        self.name = name
        self.n = n
        self.counter = [AverageMeter(name) for _ in range(n)]

    def update(self, data):
        for i in range(self.n):
            self.counter[i].update(data[i])

    def avg(self):
        return [self.counter[i].avg for i in range(self.n)]


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


def load_maskrcnn_model(args, modelfile):
    config = MitochondrionInferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.outputdir)
    model.load_weights(modelfile, by_name=True)

    return model


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


def construct_maskrcnn_pred(pred_r):
    mask = pred_r['masks']
    pred = np.zeros((mask.shape[0], mask.shape[1]))

    for i in range(mask.shape[2]):
        pred += np.array(mask[:, :, i]).astype('int32')

    pred[pred > 0] = 255

    return pred


def my_visualize(image, preds, mask, file_name, outputdir, id):
    mito_pred, er_pred, ld_pred = preds[:, :, 0], preds[:, :, 1], preds[:, :, 2]
    mito_mask, er_mask, ld_mask = mask[:, :, 1], mask[:, :, 2], mask[:, :, 3]

    mito = vis_iou(image, mito_pred, mito_mask)
    er = vis_iou(image, er_pred, er_mask)
    ld = vis_iou(image, ld_pred, ld_mask)

    vis_img = np.concatenate((mito, er, ld), axis=1)
    output_filename = os.path.join(outputdir, file_name + f'_mathch_{id}.png')
    cv2.imwrite(output_filename, vis_img)


def vis_iou(image, pred, mask):
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    image = deepcopy(image)
    lbl = np.zeros(pred.shape, dtype=np.uint8)
    match = pred.astype(np.uint8) & mask.astype(np.uint8)

    lbl = np.where(pred > 0, 1, lbl)
    lbl = np.where(mask > 0, 2, lbl)
    lbl = np.where(match == True, 3, lbl)

    colormap = np.asarray([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    lbl_viz = colormap[lbl]
    lbl_viz[lbl == 0] = (255, 255, 255)  # background

    viz = lbl_viz * 0.3 + image * 0.7

    return viz


def save_res(result, output_dir):
    TITLE = ['Mito_manual', 'Mito_pred', 'Mito_match', 'ER_manual', 'ER_pred', 'ER_match',
             'LD_manual', 'LD_pred', 'LD_match']
    d = {'File_name': []}
    d.update({c: [] for c in TITLE})
    for name in result:
        d['File_name'].append(name)
        r = result[name].avg()
        for i, t in enumerate(TITLE):
            d[t].append(r[i])

    df = pd.DataFrame.from_dict(d)
    df = df.sort_values(by=['File_name'])

    df['Mito_match_ratio'] = df['Mito_match'] / df['Mito_manual']
    df['ER_match_ratio'] = df['ER_match'] / df['ER_manual']
    df['LD_match_ratio'] = df['LD_match'] / df['LD_manual']
    df.loc[len(df.index)] = ['Mean'] + df.mean().to_list()
    output_filename = os.path.join(output_dir, 'result_match.csv')
    df.to_csv(output_filename, index=False)

    print("Result Dir:", output_filename)


def main(args):
    # 可视化后图像的文件夹
    output_dir = os.path.join(args.outputdir, 'match_vis')
    check_mkdir(output_dir)

    # Crop图片的transform
    transforms, er_transforms = get_transforms(args)

    # 读取数据用的
    dataset_val = ManualDataset()
    dataset_val.load_manual(dataset_dir=args.datadir, subset=args.subset)
    dataset_val.prepare()

    # 加载模型
    mito_model = load_maskrcnn_model(args, args.mitomodel)
    ld_model = load_maskrcnn_model(args, args.ldmodel)
    er_model = load_er_model(args)
    # ER的模型是基于catalyst框架的，需要一个runner
    device = cutils.get_device()
    myrunner = MyRunner(device=device, input_key="image", input_target_key="mask")

    # 获取每张图要随机取样的数量
    repeat_num = getattr(args, 'repeat_num', 5)
    result = {}
    try:
        for image_id in tqdm(dataset_val.image_ids):
            file_path = dataset_val.image_info[image_id]['path']
            file_name = os.path.split(file_path)[1][:-4]
            print(f'Proprocess {file_name}...')

            # 用于计算多张图算出来结果的均值
            counter = AvgCount(file_name, n=9)
            result[file_name] = counter

            # 读取图片和PM的mask
            image = dataset_val.load_image(image_id)
            mask, _ = dataset_val.load_mask(image_id)

            for i in range(repeat_num):
                try:
                    augmented = transforms(image=image, mask=mask)
                except:
                    print("====>> image size is too small: ", image.shape)
                    augmented = albu.RandomCrop(1024, 1024)(image=image, mask=mask)

                aug_image, aug_mask = augmented['image'], augmented['mask']

                # 预测Mito，tissue的数据在预测Mito的时候先Crop PM区域
                detect_image = deepcopy(aug_image)
                if args.model == 'tissue':
                    detect_image = crop_pm(detect_image, aug_mask)
                mito_pred_r = mito_model.detect([detect_image], verbose=0)[0]
                ld_pred_r = ld_model.detect([detect_image], verbose=0)[0]

                # 预测ER，需要先normalize图片，再转换成Tensor
                er_image = er_transforms(image=detect_image)['image'].unsqueeze(0)
                er_batch = {"image": er_image.to(device)}
                er_pred = predict_er(myrunner, er_model, er_batch)

                # 将Mito的结果呈现在一张图上
                mito_pred = construct_maskrcnn_pred(mito_pred_r)
                ld_pred = construct_maskrcnn_pred(ld_pred_r)

                preds = np.stack([mito_pred, er_pred * 255, ld_pred], axis=2)

                item_result = []
                s = f'|{i + 1}/{repeat_num}|{file_name}'
                for j in range(3):
                    num_match, num_organ, num_pred = matching(aug_mask[:, :, j + 1], preds[:, :, j])
                    item_result.extend([num_organ, num_pred, num_match])
                    s += f'|Match: {num_match}| Manual: {num_organ}| Pred: {num_pred}'
                print(s)

                my_visualize(detect_image, preds, aug_mask, file_name, output_dir, i)

                counter.update(item_result)
    except Exception as e:
        print("Error: ", e)
    finally:
        save_res(result, args.outputdir)
        print("Result are saved!")


if __name__ == '__main__':
    args = opts.parse_opt()

    # 每个测试放在一个单独的文件夹下
    expe_name = os.path.basename(os.path.normpath(args.datadir))
    args.outputdir = os.path.join(args.outputdir, expe_name + '_mito_er')

    print(args)
    os.environ['CUDA_VISIBLE DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # 占用GPU50%的显存
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)

    main(args)
