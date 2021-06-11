# from labelme.utils import shape_to_mask
import os, json, datetime, sys
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
print(ROOT_DIR)
from mrcnn import utils
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize

from glob import glob
from PIL import Image
from imgaug import augmenters as iaa

import matplotlib

# Agg backend runs without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


RESULTS_DIR = os.path.join(ROOT_DIR, "samples", "cell", "results")
print(RESULTS_DIR)
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class MitochondrionConfig(Config):
    BACKBONE = "resnet50"
    NAME = "Mitochondrion"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 2.0


class MitochondrionInferenceConfig(MitochondrionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.5


class MitochondrionDataset(utils.Dataset):

    def load_Mitochondrion(self, dataset_dir='data/mito_data', subset='train'):
        self.add_class("Mitochondrion", 1, "Mitochondrion")
        # assert subset in ["train", "val", "test"]
        subset_dir = subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        name_list = glob(dataset_dir + '/**/*.png', recursive=True)
        # print(name_list)
        image_ids = []
        for name in name_list:
            image_ids.append(os.path.split(name)[1][:-4])
            # image_ids.append(name.split('/')[-1].split('.')[0])
        for image_id in image_ids:
            self.add_image(
                "Mitochondrion",
                image_id=image_id,
                path=os.path.join(dataset_dir, "image/{}.png".format(image_id)))
        # name_list = glob(os.path.join(dataset_dir, '*.png'), recursive=True)
        # print(os.path.join(dataset_dir, '*.png'))
        # image_ids = []
        # for name in name_list:
        #     image_ids.append(os.path.split(name)[1][:-4])
        #     # image_ids.append(name.split('/')[-1].split('.')[0])
        # for image_id in image_ids:
        #     self.add_image(
        #         "Mitochondrion",
        #         image_id=image_id,
        #         path=os.path.join(dataset_dir, "{}.png".format(image_id)))

    def load_mask(self, image_id):
        '''
        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "mask")
        mask = []
        mask_id = info['path'].split('/')[-1].split('.')[0]
        json_data = json.load(open(os.path.join(mask_dir, str(mask_id) + '.json')))
        im = np.array(Image.open(info['path']))
        for shape in json_data['shapes']:
            rle = np.zeros(im.shape)
            enc = shape_to_mask(im.shape, points=shape['points'])
            rle[enc] = 1
            mask.append(rle)
        mask = np.stack(mask, axis=-1)
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
        '''
        # info = self.image_info[image_id]
        # mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "mask")
        # mask_id = os.path.split(info['path'])[1][:-4]
        # # mask_id = image_id
        # # mask_id = info['path'].split('/')[-1].split('.')[0]
        # mask = np.load(os.path.join(mask_dir, str(mask_id) + '.npy'))
        mask = np.zeros((1024, 1024))
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


def train():
    config = MitochondrionConfig()

    dataset_train = MitochondrionDataset()
    dataset_train.load_Mitochondrion(dataset_dir='data/cell/data/0815_Mito_MaskRCNN', subset='train')
    dataset_train.prepare()

    dataset_val = MitochondrionDataset()
    dataset_val.load_Mitochondrion(dataset_dir='data/cell/data/0815_Mito_MaskRCNN', subset='val')
    dataset_val.prepare()

    config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=DEFAULT_LOGS_DIR)

    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # weights_path = model.get_imagenet_weights()
    weights_path = COCO_WEIGHTS_PATH
    # weights_path = model.find_last()
    # weights_path = 'mito/lycode/top1+sim/mask_rcnn_mitochondrion_0039.h5'

    # model.load_weights(weights_path, by_name=True)
    model.load_weights(weights_path, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=35,
                layers='4+',
                augmentation=augmentation)

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=60,
                augmentation=augmentation,
                layers='all')


def detect():
    detect_datapath = 'data/cell/data/test/0929-cell-mito-er-test'
    # detect_datapath = 'data/tissues/data/test/0911-Mito-PM-TEM-Test'
    # detect_datapath = 'data/tissues/data/tem-mito-pm-nofangkuai'
    subset_name = 'test'
    # subset_name = 'testER0604Seg'

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "result_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    cal_dir = submit_dir + '_cal'
    cal_dir = os.path.join(RESULTS_DIR, cal_dir)
    os.makedirs(cal_dir)

    config = MitochondrionInferenceConfig()

    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=DEFAULT_LOGS_DIR)

    dataset_val = MitochondrionDataset()
    dataset_val.load_Mitochondrion(dataset_dir=detect_datapath, subset=subset_name)
    # dataset_val.load_Mitochondrion(subset='val')
    dataset_val.prepare()

    # weights_path = model.find_last()
    # weights_path = 'code/dianjing/Mask_RCNN/logs/tissue_mito_0620_100.h5'  # tissue
    weights_path = 'code/dianjing/Mask_RCNN/logs/best/cell_mito_0906_50.h5'  # cell
    # weights_path = 'code/dianjing/Mask_RCNN/logs/best/onlytem_tissue_mito_0904_50.h5'  # TEM
    # weights_path = 'code/dianjing/Mask_RCNN/logs/best/addTEM_tissue_mito_0904_50.h5'  # addTEM_tissue
    # weights_path = 'code/dianjing/C_Mask_RCNN/logs/tem-pm-nofangkuai-resnet50-20200919T1830/mask_rcnn_tem-pm-nofangkuai-resnet50-_0023.h5'

    model.load_weights(weights_path, by_name=True)

    for image_id in tqdm(dataset_val.image_ids):
        image = dataset_val.load_image(image_id)
        # gt_mask, gt_class_id = dataset_val.load_mask(image_id)
        r = model.detect([image], verbose=0)[0]

        # visualize.display_instances(
        #     image, r['rois'], r['masks'], r['class_ids'],
        #     dataset_val.class_names, r['scores'],
        #     show_bbox=False, show_mask=True,
        #     title="")
        # plt.savefig("{}/{}.png".format(submit_dir, dataset_val.image_info[image_id]["id"]))

        vis_img = visualize.drew_instances(image, r['masks']).astype(np.uint8)
        cv2.imwrite("{}/{}.png".format(submit_dir, dataset_val.image_info[image_id]["id"]), vis_img)

        mask = np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
        # print(gt_mask.shape)
        for i in range(r['masks'].shape[2]):
            tmp = np.array(r['masks'][:, :, i])
            # row = tmp.shape[0]
            # col = tmp.shape[1]
            # mask += (r['masks'][:, :, i]).astype('int32')
            mask += (tmp).astype('int32')
        image[mask < 1] = 255
        # cv2.imwrite('2.png', image)

        cv2.imwrite("{}/{}.png".format(cal_dir, dataset_val.image_info[image_id]["id"]), image)
        # cv2.imwrite('res/' + dataset_val.image_info[image_id]["id"] + '.png', image)

        # print('res/' + dataset_val.image_info[image_id]["id"] + '.png')
        # print(dataset_val.image_info[image_id]["id"], utils.compute_acc(r['masks'], gt_mask),
        #    utils.compute_recall(r['masks'], gt_mask), r['masks'].shape[2], gt_mask.shape[2])

    print("Result Dir:", submit_dir)


if __name__ == '__main__':
    # train()
    detect()
