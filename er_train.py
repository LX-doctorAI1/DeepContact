# -*- coding: utf-8 -*-
# @Time    : 2020/6/13 4:47 下午
# @Author  : 杨树鑫
# @FileName: er.py

from typing import Callable, List, Tuple
import ipdb
import os, datetime, cv2
from tqdm import tqdm
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import catalyst
from catalyst.contrib.nn import DiceLoss, IoULoss
from myutils.criterion import fscoreLoss
from catalyst.dl import utils, SupervisedRunner
from catalyst.contrib.nn import RAdam, Lookahead
import segmentation_models_pytorch as smp
from catalyst.dl.callbacks import DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback
# print(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}")

from pathlib import Path

from myutils import ERdataset
from myutils.precess import *
from myutils.visualise import *
from myutils.metric import iou
from myutils.utils import check_mkdir

SEED = 5
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True, benchmark=False)
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='训练ER模型')
    parser.add_argument('--commod', default='train',
                        help='train, eval')
    parser.add_argument('--type', default='tissue',
                        help='cell, tissue, tem')
    parser.add_argument('--datadir', default='/data/yangshuxin/data/tissues/data/tem-er-pm',
                        help='数据存放的文件夹')
    parser.add_argument('--resumefile', default='',
                        help='logs/tissue_ER_resnext101_32x8d_instagram_20200618T214138/checkpoints/best.pth')
    parser.add_argument('--logdir', default='logs',
                        help='log存放的文件夹,相对路径')
    parser.add_argument('--model', default='FPN',
                        help='模型选择，FPN or UNET')
    parser.add_argument('--encoder', default='resnext101_32x8d',
                        help='预训练模型')
    parser.add_argument('--weights', default='instagram',
                        help='预训练权重')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--encoder_learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_epochs', type=int, default=130)
    parser.add_argument('--lr_weight_decay', type=float, default=0.0003)
    parser.add_argument('--gpu', default="4,5,6,7")
    args = parser.parse_args()

    args.logdir = os.path.join(ROOT_DIR, args.logdir)
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # "" - CPU, "0" - 1 GPU, "0,1" - MultiGPU

    train_transforms = compose([
        resize_transforms(),
        hard_transforms(),
        post_transforms()
    ])
    valid_transforms = compose([pre_transforms(), post_transforms()])
    show_transforms = compose([resize_transforms(), hard_transforms()])

    ENCODER_WEIGHTS = args.weights  # 'instagram'
    ENCODER_NAME = args.encoder  # 'resnext101_32x8d'
    # model = smp.FPN(encoder_name="resnext101_32x8d", encoder_weights=ENCODER_WEIGHTS, classes=1)
    # model = smp.Unet(encoder_name="resnext101_32x8d", encoder_weights=ENCODER_WEIGHTS, classes=1)
    # model = smp.FPN(encoder_name="se_resnext101_32x4d", encoder_weights=ENCODER_WEIGHTS, classes=1)
    if args.model == 'FPN':
        model = smp.FPN(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, classes=1)
    elif args.model == "UNET":
        model = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, classes=1)
    elif args.model == "linknet":
        model = smp.Linknet(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, classes=1)
    elif args.model == "pspnet":
        model = smp.PSPNet(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, classes=1)
    elif args.model == "deeplab":
        model = smp.DeepLabV3(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, classes=1)

    device = utils.get_device()
    print(f"device: {device}")
    # by default SupervisedRunner uses "features" and "targets",
    # in our case we get "image" and "mask" keys in dataset __getitem__
    runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")
    ROOT = Path(args.datadir)
    if args.commod == 'train':
        train_image_path = ROOT / "train/image"
        train_mask_path = ROOT / "train/mask"

        ALL_IMAGES = sorted(train_image_path.glob("*.png"))
        ALL_MASKS = sorted(train_mask_path.glob("*.npy"))
        # print(ALL_IMAGES[:10])

        loaders = ERdataset.get_loaders(
            images=ALL_IMAGES,
            masks=ALL_MASKS,
            random_state=SEED,
            train_transforms_fn=train_transforms,
            valid_transforms_fn=valid_transforms,
            batch_size=args.batch_size
        )

        # we have multiple criterions
        criterion = {
            "dice": DiceLoss(),
            "iou": IoULoss(),
            "bce": nn.BCEWithLogitsLoss(),
            "fscore": fscoreLoss(),
        }
        if args.resumefile:
            # Since we use a pre-trained encoder, we will reduce the learning rate on it.
            checkpoint = torch.load(args.resumefile)
            # layerwise_params = checkpoint
            print("===========>>reload model stats dict from ", args.resumefile)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("===========>>reloaded model stats dict!")

        layerwise_params = {"encoder*": dict(lr=args.encoder_learning_rate, weight_decay=args.lr_weight_decay)}

        # This function removes weight_decay for biases and applies our layerwise_params
        model_params = utils.process_model_params(model, layerwise_params=layerwise_params)

        # ipdb.set_trace()
        # Catalyst has new SOTA optimizers out of box
        base_optimizer = RAdam(model_params, lr=args.learning_rate, weight_decay=args.lr_weight_decay)
        optimizer = Lookahead(base_optimizer)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=10)

        logdir = os.path.join(args.logdir,
                              "{}_{}_{}_{}_{:%Y%m%dT%H%M}".format(args.type, args.model, ENCODER_NAME, ENCODER_WEIGHTS, datetime.datetime.now()))
        print("=======>> log dir: ", logdir)
        check_mkdir(logdir)

        callbacks = [
            # Each criterion is calculated separately.
            CriterionCallback(
                input_key="mask",
                prefix="loss_dice",
                criterion_key="dice"
            ),
            CriterionCallback(
                input_key="mask",
                prefix="loss_iou",
                criterion_key="iou"
            ),
            CriterionCallback(
                input_key="mask",
                prefix="loss_bce",
                criterion_key="bce"
            ),
            CriterionCallback(
                input_key="mask",
                prefix="loss_fscore",
                criterion_key="fscore"
            ),

            # And only then we aggregate everything into one loss.
            MetricAggregationCallback(
                prefix="loss",
                mode="weighted_sum", # can be "sum", "weighted_sum" or "mean"
                # because we want weighted sum, we need to add scale for each loss
                metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8, "loss_fscore": 1.0},
            ),

            # metrics
            DiceCallback(input_key="mask"),
            IouCallback(input_key="mask"),
        ]

        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            # resume=args.resumefile,
            # our dataloaders
            loaders=loaders,
            # We can specify the callbacks list for the experiment;
            callbacks=callbacks,
            # path to save logs
            logdir=logdir,
            num_epochs=args.num_epochs,
            # save our best checkpoint by IoU metric
            main_metric="iou",
            # IoU needs to be maximized.
            minimize_metric=False,
            # for FP16. It uses the variable from the very first cell
            fp16=None,
            # prints train logs
            verbose=True,
        )
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
