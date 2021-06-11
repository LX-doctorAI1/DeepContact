import os, glob, datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import sys
import time
import numpy as np
import cv2
from tqdm import tqdm
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Agg backend runs without a display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# import warnings
# warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
RESULTS_DIR = os.path.join(ROOT_DIR, "samples", "cell", "results")


############################################################
#  Configurations
############################################################


class MitochondrionConfig(Config):
    BACKBONE = "resnet50"
    NAME = "Mitochondrion"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 2.0


class MitochondrionInferenceConfig(MitochondrionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.5


class MitoDataset(utils.Dataset):

    def load_Mito(self, dataset_dir='data/mito_data', subset='train', class_ids=None):
        self.add_class("ict", 1, "Mito")
        assert subset in ["train", "val", "DL_C_UNET"]
        subset_dir = subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        name_list = glob.glob(dataset_dir + '/**/*.tif', recursive=True)
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

    # def load_image(self, image_id):
    #     """Load the specified image and return a [H,W,3] Numpy array.
    #     """
    #     import skimage
    #     # Load image
    #     image = skimage.io.imread(self.image_info[image_id]['path'])
    #     # If grayscale. Convert to RGB for consistency.
    #     if image.ndim != 3:
    #         image = skimage.color.gray2rgb(image)
    #     # If has an alpha channel, remove it for consistency
    #     if image.shape[-1] == 4:
    #         image = image[..., 0]
    #     return image

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
        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "mask")
        mask_id = os.path.split(info['path'])[1][:-4]
        # mask_id = image_id
        # mask_id = info['path'].split('/')[-1].split('.')[0]
        mask = np.load(os.path.join(mask_dir, str(mask_id) + '.npy'))
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Mito.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)


    # Configurations
    if args.command == "train":
        config = MitochondrionConfig()
    else:
        class InferenceConfig(MitochondrionConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    model_path = args.model
    # model_path = args.model'mito/lycode/top1+sim/mask_rcnn_mitochondrion_0039.h5'

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)
    # if args.command == "train":
    #     model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    #     # model.load_weights(model_path, by_name=True)
    # else:
    #     model.load_weights(model_path, by_name=True, exclude=[])

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = MitochondrionDataset()
        dataset_train.load_Mitochondrion(dataset_dir=args.dataset, subset='train')
        dataset_train.prepare()

        dataset_val = MitochondrionDataset()
        dataset_val.load_Mitochondrion(dataset_dir=args.dataset, subset='val')
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        # augmentation = imgaug.augmenters.Fliplr(0.5)
        augmentation = iaa.SomeOf((0, 2), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180),
                       iaa.Affine(rotate=270)]),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0))
        ])

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=60,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=80,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=120,
                    layers='all',
                    augmentation=augmentation)

    else:
        detect_datapath = args.dataset
        # detect_datapath = 'data/tissues/testER0604Seg03/'
        subset_name = 'DL_C_UNET'

        # 保存预测结果和用于计算cal的结果
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
        dataset_val.prepare()

        # weights_path = model.find_last()
        weights_path = args.model
        # weights_path = 'mito/lycode/top1+sim/mask_rcnn_mitochondrion_0039.h5'
        model.load_weights(weights_path, by_name=True)

        for image_id in tqdm(dataset_val.image_ids):
            image = dataset_val.load_image(image_id)
            gt_mask, gt_class_id = dataset_val.load_mask(image_id)
            r = model.detect([image], verbose=0)[0]
            # from fuc import calc, extract_bboxes, compute_ap_range
            # gt_box = utils.extract_bboxes(gt_mask)
            # visualize.display_instances(
            #    image, gt_box, gt_mask, gt_class_id,
            #    dataset_val.class_names, None,
            #    show_bbox=False, show_mask=True,
            #    title="")
            visualize.display_instances(
                image, r['rois'], r['masks'], r['class_ids'],
                dataset_val.class_names, r['scores'],
                show_bbox=False, show_mask=True,
                title="")
            plt.savefig("{}/{}.png".format(submit_dir, dataset_val.image_info[image_id]["id"]))

            mask = np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
            # print(gt_mask.shape)
            for i in range(r['masks'].shape[2]):
                tmp = np.array(r['masks'][:, :, i])
                mask += (tmp).astype('int32')
            image[mask < 1] = 255

            cv2.imwrite("{}/{}.png".format(cal_dir, dataset_val.image_info[image_id]["id"]), image)

