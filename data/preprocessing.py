import numpy as np
import timm
import torch

from abc import ABC, abstractmethod

from PIL import Image
from torchvision import transforms


class Preprocessing(ABC):
    @abstractmethod
    def __call__(self, image_data, decode_bytes=False, augment=False):
        pass


class ResNetImagePreprocessing(Preprocessing):
    """Preprocesses images to be input to a model. ResNet version.

    Resnet style preprocessing: scale so the smallest side is 256, center crop to a 227x227 square, subtract the
    ImageNet mean and divide by the ImageNet variance.
    """

    DEFAULT_IMAGE_SIZE = 227
    DEFAULT_ALIGN_SMALL_EDGE_SIZE = 256
    DEFAULT_MEAN_RGB = [0.485, 0.456, 0.406]
    DEFAULT_STD_RGB = [0.229, 0.224, 0.225]
    DEFAULT_HORIZONTAL_FLIP_PROBABILITY = 0.5

    def __init__(self, _, args):
        """Create a ResNetImagePreprocessing object that uses the specified parameters."""

        self.image_size = self.DEFAULT_IMAGE_SIZE if args.image_size is None else args.image_size

        # Make the three chain of transforms for the plain, augmentation and canonical modes.
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.DEFAULT_ALIGN_SMALL_EDGE_SIZE),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.DEFAULT_MEAN_RGB, std=self.DEFAULT_STD_RGB),
            ]
        )
        self.transforms_augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(self.DEFAULT_HORIZONTAL_FLIP_PROBABILITY),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.DEFAULT_MEAN_RGB, std=self.DEFAULT_STD_RGB),
            ]
        )
        self.transforms_canonical = transforms.Compose(
            [transforms.Resize(self.DEFAULT_ALIGN_SMALL_EDGE_SIZE), transforms.CenterCrop(self.image_size)]
        )

    def __call__(self, image_data, decode_bytes=False, augment=False):
        # Convert from NumPy array to PIL image, if requested.
        if decode_bytes:
            image_data = Image.fromarray(image_data)

        # Apply the chain of transforms to the PIL Image version of the data.
        if augment:
            return self.transforms_augmentation(image_data)
        return self.transforms(image_data)

    def get_canonical(self, image_data, decode_bytes=True):
        # Convert from NumPy array to PIL image, if requested.
        if decode_bytes:
            image_data = Image.fromarray(image_data)

        return torch.tensor(np.array(self.transforms_canonical(image_data)))

    def get_preprocessed_type(self):
        """Get the type of the preprocessed image."""

        return torch.float32

    def get_preprocessed_shape(self):
        return [self.image_size, self.image_size, 3]

    def get_decoded_input_type(self):
        """Get the type of the decoded image."""

        return np.uint8


class ViTImagePreprocessing(Preprocessing):
    """Preprocess images to be input to a model. ViT version."""

    DEFAULT_IMAGE_SIZE = 224

    def __init__(self, _, args):
        """Create a ViTImagePreprocessing object that uses the specified parameters."""

        self.image_size = self.DEFAULT_IMAGE_SIZE if args.image_size is None else args.image_size

        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)

        config = timm.data.resolve_data_config({}, model=model)
        self.transforms = timm.data.transforms_factory.create_transform(**config)
        self.transforms_augmentation = timm.data.transforms_factory.create_transform(**{**config, "is_training": True})

    def __call__(self, image_data, decode_bytes=False, augment=False):
        # Convert from NumPy array to PIL image, if requested.
        if decode_bytes:
            image_data = Image.fromarray(image_data)

        # Apply the chain of transforms to the PIL Image version of the data.
        if augment:
            return self.transforms_augmentation(image_data)
        return self.transforms(image_data)

    def get_preprocessed_type(self):
        """Get the type of the preprocessed image."""

        return torch.float32

    def get_preprocessed_shape(self):
        return [self.image_size, self.image_size, 3]

    def get_decoded_input_type(self):
        """Get the type of the decoded image."""

        return np.uint8


class IndexRemapPreprocessing(Preprocessing):
    def __init__(self, dataset_builder, _):
        self.index_remapping = torch.tensor(dataset_builder.get_instance_index_remapping(), dtype=torch.int32)

    def __call__(self, data, decode_bytes=True, augment=False):
        i = self.index_remapping[data]
        return i

    def get_preprocessed_type(self):
        return torch.int32

    def get_preprocessed_shape(self):
        return ()

    def get_decoded_input_type(self):
        return torch.int32
