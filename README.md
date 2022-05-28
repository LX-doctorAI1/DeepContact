# DeepContact

**DeepContact software** is the tensorflow/keras and Pytorch implementation for high throughput quantification of membrane contact site. This repository is developed based on the under review paper [**DeepContact: High throughput quantification of membrane contact site based on electron microscopy imaging**](https://).<br>

Author: Liqing Liu<sup>1†</sup>, Shuxin Yang<sup>2,5†</sup>, Yang Liu<sup>2,5</sup>, Junjie Hu<sup>1,4*</sup>, Li Xiao<sup>2,5,6*</sup> and Tao Xu<sup>1,3,4*</sup>

<sup>1</sup>National Laboratory of Biomacromolecules, Institute of Biophysics, Chinese Academy of Sciences, Beijing, China. \
<sup>2</sup>Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.\
<sup>3</sup>Bioland Laboratory (Guangzhou Regenerative Medicine and Health Guangdong Laboratory), Guangzhou, Guangdong, China.\
<sup>4</sup>College of Life Science, University of Chinese Academy of Sciences, Beijing, China.\
<sup>5</sup>School of Computer and Control Engineering, University of Chinese Academy of Sciences, Beijing, China.\
<sup>6</sup>Ningbo HuaMei Hospital, University of Chinese Academy of Sciences, Ningbo, China.\
<sup>†</sup>These authors contributed equally: Liqing Liu, Shuxin Yang. \
<sup>*</sup>Correspondence to: xutao@ibp.ac.cn; xiaoli@ict.ac.cn; huj@ibp.ac.cn.

## Abstract
> Membrane contact site (MCS)-mediated organelle interactions plays essential roles in the cell. Quantitative analysis of the MCS reveals vital clues for cellular responses under various physiological and pathological conditions. However, an efficient tool is yet lacking. Here, we developed “DeepContact”, a deep learning protocol for optimizing organelle segmentation and contact analysis based on label-free electron microscopy (EM). DeepContact presents high efficiency and flexibility in interactive visualizations, accommodating new morphologies of organelles and recognizing contacts in versatile width ranges, which enables statistical analysis of various types of MCSs in multiple systems. DeepContact revealed the importance in keeping native topological organization. Furthermore, it profiled previously unidentified coordinative rearrangements of MCS types in combined nutritional conditionings of cultured cells. DeepContact also unveiled a subtle wave of ER-mitochondrial entanglement in the Sertoli cell during the seminiferous epithelial cycle, indicating its potential in bridging MCS dynamics to physiological and pathological processes. 

## Contents
- [Environment](#environment)
- [File structure](#file-structure)
- [Test pre-trained models](#test-pre-trained-models)
- [Train a new model](#train-a-new-model)
- [License](#License)
- [Citation](#citation)

## Environment
- Ubuntu 16.04
- CUDA 9.0
- cuDNN 7.0
- Python 3.6.10
- Tensorflow 1.5.0
- Keras 2.0.8
- segmentation-models-pytorch 0.1.0
- catalyst 20.5.1
- GPU: >= GeForce GTX 1080Ti
- pycococreator:(only for training) 
    ```
    pip install -r requirements.txt
    pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
    ```

## File structure
- `./checkpoint`: place pre-trained mitochondrial/er models here for testing
- `./config`: the config file for testing
- `./dataset`: the default path for training data and testing data
    - `./dataset/mito_train` The augmented data for training mitochondrial model. It follows MS COCO format.
    - `./dataset/er_train` The augmented data for training er model.
- `./DeeContact_Amira`: DeepContact's extension for Amira 3D software
- `./mrcnn`: the adaptive MaskRCNN source code. The mitochondrial model is adapted from MaskRCNN.
- `./myutils`: util functions for DeepContact project.
- `./samples`: training for mitochondrial data.
    - `coco/coco.py`:  training a new data: 
    ```
    python coco.py train --dataset=/path/to/mito_data/ --model=coco
    ```
- `./add_er.py`: preprocess er data for training. Details see the Args in the file.
- `./add_mito.py`: preprocess mitochondrial data for training. Details see the Args in the file.
- `./contact.py`: calculate contact between mitochondrial and er.
- `./er_train.py`: training for mitochondrial data. Details see the Args in the file.
- `./main_predict_mito_er_10px.py`: analysis the membrane contact site between mitochondrial and er
    - method: 
    ```
    python main_predict_mito_er_10px.py --cfg=config/cell_mito_er.yml -d=/path/to/data/ --gpu=0
    ```
- `./main_predict_mito_ld_10px.py`: analysis the membrane contact site between mitochondrial and ld
    - method: 
    ```
    python main_predict_mito_ld_10px.py --cfg=config/cell_mito_ld.yml -d=/path/to/data/ --gpu=0
    ```
- `./precess.py`: util functions for DeepContact project.

## Test pre-trained models
- Download [pre-trained models](https://doi.org/10.6084/m9.figshare.19845940.v1) of Mitochondrial/ER model and place them in `./checkpoint/`
- Open your terminal and cd to `deepcontact`
- Run 
    ```
    python main_predict_mito_er_10px.py --cfg=config/cell_mito_er.yml -d=/path/to/data/ --gpu=0
    ```
 in your terminal. Note that before running the bash file, you should check if the data paths and other arguments are set correctly
- The output images will be saved in `./results`
- Typical results: (left:mitochondrial, middle: ER,  right: Contact, the original image can be found in `./dataset/U2OS1_26.tif`)
![Results](figures/U2OS1_26.png)
<!-- <br>
<p align="center"><img width="800" src="figures/U2OS1_26.png"></p> -->

## Train a new model
- Data for training: You can train a new DeepContact model using [microscopy imaging](https://doi.org/10.6084/m9.figshare.19898404.v1) or your own datasets. Note that you'd better divide the dataset of each specimen into training part and validation/testing part before training, so that you can test your model with the preserved validation/testing data
- Data preprocess: run `./add_er.py` and `./add_mito.py` to creat image patch pairs of datasets. Before running, you should check image paths and some parameters following the instructions in `./add_er.py` and `./add_mito.py`. After running, the augumented data is saved in `./dataset/` by default
- DeepContact's Mitochondrion model:
    - Run `python ./sample/coco/coco.py train --dataset=/path/to/mito_data/ --model=coco` in your terminal to train a new DeepContact's Mitochondrion model. Similar to testing, before running the bash file, you should check if the data paths and the arguments are set correctly
    - Run ``python er_train.py --commod=train --type=cell --datadir=/path/to/er_data/ --gpu=0`` in your terminal to train a new DeepContact's ER model. Similar to testing, before running the bash file, you should check if the data paths and the arguments are set correctly
- You can run `tensorboard --logdir [save_weights_dir]` to monitor the training process via tensorboard. If the validation loss isn't likely to decay any more, you can use early stop strategy to end the training
- Model weights will be saved in `./logs` by default

## License
This repository is released under the MIT License (refer to the LICENSE file for details).

Please contact shuxin Yang(aspenstarss@gmail.com) or Li Xiao(andrew.lxiao@gmail.com) if you have any problem with the codes.

## Citation
If you find the code or dataset helpful in your resarch, please cite the following paper:
```
@article{Liu2022deepcontact,
  title={DeepContact: High throughput quantification of membrane contact site based on electron microscopy imaging},       
  author={Liqing Liu, Shuxin Yang, Yang Liu, Junjie Hu, Li Xiao and Tao Xu},
  journal={Journal of Cell Biology},
  pages={},
  year={2022},
  publisher={Rockefeller University Press}
}
```
