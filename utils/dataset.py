import os
from glob import glob
import numpy as np


from mrcnn import utils
from utils.utils import labelme_load_mask_one


class MitochondrionDataset(utils.Dataset):

    def load_Mitochondrion(self, dataset_dir='data/mito_data', subset=''):
        self.add_class("Mitochondrion", 1, "Mitochondrion")
        # assert subset in ["train", "val", "test"], 'subset must in ["train", "val", "test"]'
        dataset_dir = os.path.join(dataset_dir, subset)
        name_list = glob(dataset_dir + '*.png', recursive=True) + \
                    glob(dataset_dir + '*.jpg', recursive=True) + \
                    glob(dataset_dir + '*.tif', recursive=True)

        for name in name_list:
            image_id = os.path.split(name)[1]
            self.add_image("Mitochondrion", image_id=image_id,
                           path=os.path.join(dataset_dir, image_id))

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        json_data = os.path.splitext(info['path'])[0] + '.json'

        image = self.load_image(image_id)
        img_shape = image.shape[:-1]

        if os.path.exists(json_data):
            mask = labelme_load_mask_one(json_data, ['Plasma Membrane'], img_shape)
            if np.sum(mask) == 0:
                mask = np.ones(img_shape)
        else:
            mask = np.ones(img_shape)

        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


class ManualDataset(utils.Dataset):

    def load_manual(self, dataset_dir='', subset=''):
        self.add_class("Manual", 1, "Manual")
        dataset_dir = os.path.join(dataset_dir, subset)
        name_list = glob(dataset_dir + '*.png', recursive=True) + \
                    glob(dataset_dir + '*.jpg', recursive=True) + \
                    glob(dataset_dir + '*.tif', recursive=True)

        for name in name_list:
            image_id = os.path.split(name)[1]
            self.add_image("Manual", image_id=image_id,
                           path=os.path.join(dataset_dir, image_id))

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        json_data = os.path.splitext(info['path'])[0] + '.json'

        image = self.load_image(image_id)
        img_shape = image.shape[:-1]

        if os.path.exists(json_data):
            pm = labelme_load_mask_one(json_data, ['Plasma Membrane'], img_shape)
            if pm is None or pm.sum() == 0:
                pm = np.ones(img_shape)
            mito = labelme_load_mask_one(json_data, ['Mito'], img_shape)
            er = labelme_load_mask_one(json_data, ['ER'], img_shape)
            ld = labelme_load_mask_one(json_data, ['LD'], img_shape)

            mask = np.stack([pm, mito, er, ld], axis=2)
        else:
            mask = np.ones((img_shape + (4,)))

        return mask, np.ones([mask.shape[-1]], dtype=np.int32)