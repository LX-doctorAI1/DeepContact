from mrcnn.config import Config

class MitochondrionConfig(Config):
    BACKBONE = "resnet101"
    # BACKBONE = "resnet50"
    NAME = "Mitochondrion"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    # IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    # IMAGE_MIN_SCALE = 2.0


class MitochondrionInferenceConfig(MitochondrionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # IMAGE_RESIZE_MODE = "pad64"
    # RPN_NMS_THRESHOLD = 0.7
    DETECTION_MIN_CONFIDENCE = 0.0
    DETECTION_NMS_THRESHOLD = 0.3


class LDConfig(Config):
    #BACKBONE = "resnet101"
    BACKBONE = "resnet50"
    NAME = "Mitochondrion"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    # IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    # IMAGE_MIN_SCALE = 2.0


class LDInferenceConfig(LDConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # IMAGE_RESIZE_MODE = "pad64"
    # RPN_NMS_THRESHOLD = 0.7
    DETECTION_MIN_CONFIDENCE = 0.0
    DETECTION_NMS_THRESHOLD = 0.3


