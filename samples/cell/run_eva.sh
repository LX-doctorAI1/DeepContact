#!/bin/bash
echo "Start"
#python coco.py evaluate --dataset=data/tissues/data/mito-pm --model=code/dianjing/C_Mask_RCNN/logs/lymodel-tissue-tem-resnet5020200913T1429/mask_rcnn_lymodel-tissue-tem-resnet50_0050.h5
#CUDA_VISIBLE_DEVICES=5 python coco.py evaluate --dataset=data/tissues/data/tem-mito  --model=code/dianjing/cmaskrcnn/logs/tissue-tem-resnet10120201006T2204/mask_rcnn_tissue-tem-resnet101_0050.h5
#CUDA_VISIBLE_DEVICES=5 python coco.py evaluate --dataset=data/cell/data/ld-1004-down  --model=code/dianjing/cmaskrcnn/logs/cell-ld-down-resnet5020201005T0114/mask_rcnn_cell-ld-down-resnet50_0050.h5
#CUDA_VISIBLE_DEVICES=3 python coco.py evaluate --dataset=data/cell/data/1014-cell-mito  --model=logs/cell-mito-resnet10120201014T2228/mask_rcnn_cell-mito-resnet101_0050.h5
#CUDA_VISIBLE_DEVICES=5 python coco.py evaluate --dataset=data/cell/data/1014-cell-mito  --model=checkpoint/cell_mito.h5
#CUDA_VISIBLE_DEVICES=3 python coco.py evaluate --dataset=data/cell/data/cell-mito-downsample  --model=logs/cell-mito-resnet10120201029T1106/mask_rcnn_cell-mito-resnet101_0050.h5
#CUDA_VISIBLE_DEVICES=3 python coco.py evaluate --dataset=data/cell/data/cell-mito-downsample  --model=logs/cell-mito-resnet10120201027T1026/mask_rcnn_cell-mito-resnet101_0028.h5
#CUDA_VISIBLE_DEVICES=5 python coco.py evaluate --dataset=data/cell/data/1014-cell-mito  --model=logs/cell-mito-resnet10120201107T0800/mask_rcnn_cell-mito-resnet101_0026.h5
#CUDA_VISIBLE_DEVICES=6 python coco.py evaluate --dataset=data/cell/data/cell-ld-downsample  --model=logs/cell-ld-resnet10120201107T0821/mask_rcnn_cell-ld-resnet101_0030.h5 
CUDA_VISIBLE_DEVICES=6 python coco.py evaluate --dataset=data/cell/data/cell-mito-merge-downsample  --model=logs/cell-mito-resnet10120201107T0800/mask_rcnn_cell-mito-resnet101_0026.h5 
echo "End"
