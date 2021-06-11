## DeepContact

## Requirements
    - Python 3
    - Pytorch 1.7
    - TensorFlow 1.5.0

### Code structure
- Config: the model's config argparses during detection.
- Datasets: the base class of UNet model used for ER.
- mrnn & sample: the code of Mask R-CNN used for Mito and LD.
- utils: used for criterion, metric and visualization.
- contact: used for calculating the contacts between Mito and ER/LD.
- main_match & organ_match: used for calculating the ratio of overlap greate than 0.6 between detection and labeled organ.
- main: used for detection.

## Training
The Mito training codes locate in samples, e.g., `run_train.py`
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python coco.py train --ngpu=4 --name=cell-mito-resnet101 --dataset=data/cell/data/cell-mito-merge-downsample --model=checkpoint/cell_mito.h5
```

The ER training code is `er-train.py`.
```
python er-train.py train --type {$cell/tissue} --datadir {$DATA_DIR} --model UNET --batch_size {$BATCHSIZE} --learning_rate {$LR} --num_epochs {$EPOCH} 
```

## Analysis
```
python main_mito_{er/ld}.py --cfg config/{$config_file} --gpu {$GPU_IDs} --datadir {$DATA_DIR} --outputdir {$SAVEDIR} --resolution {Tissue=10, Cell=5} 
```
Please contact Li Xiao(andrew.lxiao@gmail.com) or Shuxin Yang(yangshuxin19g@ict.ac.cn) for any problem with the code.