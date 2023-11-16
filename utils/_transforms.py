from dataclasses import dataclass
from random import choice, random
from torch import device
from torchvision import transforms


@dataclass
class TransformConfig:
    insize: tuple[int] = (64, 64)
    cropsize: tuple[int] = (256, 256)
    outsize: tuple[int] = (32, 32)
    device = device("cuda")

    def __post_init__(self):
        if isinstance(self.size, int):
            self.size = (self.size, self.size)


train_transform = transforms.Compose(
    [
        transforms.Resize(TransformConfig.insize),
        # transforms.RandomResizedCrop(TransformConfig.cropsize),
        transforms.RandomRotation(10),
        # transforms.Resize(TransformConfig.outsize),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=choice([3, 5, 7]))], p=0.5),
        transforms.ToTensor(),
    ]
)


val_transform = transforms.Compose(
    [
        transforms.Resize(TransformConfig.insize),
        transforms.RandomResizedCrop(TransformConfig.cropsize),
        transforms.RandomRotation(30),
        transforms.Resize(TransformConfig.outsize),
        transforms.ToTensor(),
    ]
)
