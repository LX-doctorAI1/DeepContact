from _hx_core import hx_message

from enum import Enum

class DeepLearningModule(Enum):
    TRAINING = 1
    PREDICTION = 2

class DeepLearningEnvironmentStatus(Enum):
    OK = 0
    PYTHON_ENV_LOADING_ERROR = 1
    MATLAB_NOT_AVAILABLE = 2
    NOT_LOADED = -1

def lock():
    import ctypes
    mcr_locker_so = ctypes.cdll.LoadLibrary("matlabmcrlocker.dll")
    tryLock = mcr_locker_so.McrLocker_tryContext
    tryLock.restype = ctypes.c_bool
    return tryLock(ctypes.c_int(2))

def _get_module_name(module_type):
    if module_type == DeepLearningModule.TRAINING:
        return "training"
    elif module_type == DeepLearningModule.PREDICTION:
        return "prediction"
    return ""

def get_error_str(error, module_type):
    if DeepLearningEnvironmentStatus.PYTHON_ENV_LOADING_ERROR:
        return ("Loading the Python environment for the " + _get_module_name(module_type) + " module failed.\n"
               "Please use the Python menu to install the Deep Learning environment.\n"
               "Furthermore, make sure your GPU compute capability is 3.5 or higher,\n"
               "your environment doesn't set a PYTHONPATH environment variable pointing\n"
               "to a Python installation nor set a CUDA installation path in the PATH\n"
               "or LD_LIBRARY_PATH environment variables, your graphics drivers are\n"
               "up to date, and your CPU supports AVX2 extensions.")
    elif DeepLearningEnvironmentStatus.MATLAB_NOT_AVAILABLE:
        return ("It is not possible to launch DeepLearning modules \n"
        "if Calculus or GenerateTracks have already been instanciated.\n\n"
        "Please restart a new session.\n")
    return ""

def check_environment():
    # if not lock():
    #     return DeepLearningEnvironmentStatus.MATLAB_NOT_AVAILABLE

    try:
        import cv2
        from copy import deepcopy
        import numpy as np
        import warnings
        warnings.filterwarnings('ignore')
        import skimage
        import time

        import torch
        import tensorflow as tf

        from catalyst.dl import SupervisedRunner
        from catalyst.dl import utils as cutils
        import segmentation_models_pytorch as smp
        import albumentations as albu

        from mrcnn import model as modellib
        from mrcnn import visualize

        from contact import calContactDist_er_elongation
        from precess import label2rgb
        from config.mrcnn_config import MitochondrionInferenceConfig

    # In Python 3.6 ModuleNotFoundError exception could be used to provide a more specific error message to the user
    except ImportError as e:
        return DeepLearningEnvironmentStatus.PYTHON_ENV_LOADING_ERROR

    return DeepLearningEnvironmentStatus.OK
