#!/bin/bash
echo "Start"
#python coco.py train --dataset=data/tissues/data/mito-pm  --model=code/dianjing/C_Mask_RCNN/logs/lymodel-tissue-tem-resnet5020200913T1429/mask_rcnn_lymodel-tissue-tem-resnet50_0050.h5
#CUDA_VISIBLE_DEVICES=4,5,6,7 python coco.py train --ngpu=4 --name=cell-mito-resnet101 --dataset=data/cell/data/cell-mito-downsample --model=cell-mito-resnet10120201030T1005/mask_rcnn_cell-mito-resnet101_0017.h5
CUDA_VISIBLE_DEVICES=0,1,2,3 python coco.py train --ngpu=4 --name=cell-mito-resnet101 --dataset=data/cell/data/cell-mito-merge-downsample --model=checkpoint/cell_mito.h5
#CUDA_VISIBLE_DEVICES=4,5,6,7 python coco.py train --ngpu=4 --name=cell-mito-resnet101 --dataset=data/cell/data/cell-mito-merge-downsample --model=coco
#CUDA_VISIBLE_DEVICES=0,1,2,3 python coco.py train --ngpu=4 --name=cell-ld-resnet101 --dataset=data/cell/data/cell-mito-downsample --model=cell-mito-resnet10120201102T1049/mask_rcnn_cell-mito-resnet101_0016.h5
#CUDA_VISIBLE_DEVICES=6 python coco.py train --ngpu=1 --name=tem+pm-mito-resnet101 --dataset=data/tissues/data/tem-mito-pm --model=coco
echo "End"

#!/bin/bash
echo "Start"
#python coco.py train --dataset=data/tissues/data/mito-pm  --model=code/dianjing/C_Mask_RCNN/logs/lymodel-tissue-tem-resnet5020200913T1429/mask_rcnn_lymodel-tissue-tem-resnet50_0050.h5
#CUDA_VISIBLE_DEVICES=4,5,6,7 python coco.py train --ngpu=4 --name=cell-mito-resnet101 --dataset=data/cell/data/cell-mito-downsample --model=cell-mito-resnet10120201030T1005/mask_rcnn_cell-mito-resnet101_0017.h5
CUDA_VISIBLE_DEVICES=0,1,2,3  python coco.py train --ngpu=4 --name=cell-ld-resnet101 --dataset=data/cell/data/cell-ld-downsample --model=cell-ld-resnet10120201107T0821/mask_rcnn_cell-ld-resnet101_0016.h5
#CUDA_VISIBLE_DEVICES=0,1,2,3 python coco.py train --ngpu=4 --name=cell-ld-resnet101 --dataset=data/cell/data/cell-mito-downsample --model=cell-mito-resnet10120201102T1049/mask_rcnn_cell-mito-resnet101_0016.h5
#CUDA_VISIBLE_DEVICES=6 python coco.py train --ngpu=1 --name=tem+pm-mito-resnet101 --dataset=data/tissues/data/tem-mito-pm --model=coco
echo "End"