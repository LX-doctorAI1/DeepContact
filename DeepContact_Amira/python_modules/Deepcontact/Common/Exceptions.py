from Invokable import Invokable

class CouldNotWriteArchitectureFile(Invokable.Exception):
    def __init__(self, invokable_name, filename):
        super().__init__("An error occurred: unable to write the model architecture file to disk: " + filename, invokable_name)

class TrainingSetTooSmall(Invokable.Exception):
    def __init__(self, invokable_name, fake_invokable_name):
        super().__init__("Invalid input: not enough samples in the training set.", fake_invokable_name)

class ValidationSetTooSmall(Invokable.Exception):
    def __init__(self, invokable_name, fake_invokable_name):
        super().__init__("Invalid input: not enough samples in the validation set.", fake_invokable_name)

class CouldNotWriteWeightsFile(Invokable.Exception):
    def __init__(self, invokable_name, filename):
        super().__init__("An error occurred: unable to write the model weights file to disk: " + filename + ". Model weights may have been saved in the checkpoint folder.", invokable_name)

class CouldNotWriteFinalWeightsFile(Invokable.Exception):
    def __init__(self, invokable_name, filename):
        super().__init__("An error occurred: unable to write the model weights file to disk: " + filename + ".", invokable_name)

class InvalidWeightsFile(Invokable.Exception):
    def __init__(self, invokable_name):
        super().__init__("Invalid input: initial weights file incompatible with the selected model architecture.", invokable_name)

class WeightsFileNotFound(Invokable.Exception):
    def __init__(self, invokable_name):
        super().__init__("Invalid input: initial weights file not found.", invokable_name)

class NoOuputDirectorySet(Invokable.Exception):
    def __init__(self, invokable_name):
        super().__init__("Invalid input: no output directory path set.", invokable_name)

class OutputDirectoryNotFound(Invokable.Exception):
    def __init__(self, invokable_name):
        super().__init__("Invalid parameter: output directory path not found.", invokable_name)

class ImageSizesDoNotMatch(Invokable.Exception):
    def __init__(self, invokable_name, mask_shape, image_shape):
        super().__init__("Invalid input: the size of the target image (" + str(mask_shape[0]) + ", " + str(mask_shape[1]) + ") does not match that of the input image (" + str(image_shape[0]) + ", " + str(image_shape[1]) +").", invokable_name)

class CannotOpenImage(Invokable.Exception):
    def __init__(self, invokable_name):
        super().__init__("An error occurred: error accessing an image from the Data Collection.", invokable_name)

class NoPermission(Invokable.Exception):
    def __init__(self, invokable_name):
        super().__init__("The user has no permissions to modify the specific folder filled in (writing especially). Change the user permissions or enter a new folder with writing rights.", invokable_name)
