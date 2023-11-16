import random
from math import floor
from multiprocessing import cpu_count
from .lmdb import ImageFolderLMDB
import torch.utils.data as torchdata
import torchvision
import torchvision.transforms as transforms


def imagedataloader(split: tuple[float],
                    path: str,
                    batchsize: int = 16,
                    transform: transforms.Compose = transforms.ToTensor(),
                    threads: int = cpu_count()
                    ) -> tuple[torchdata.DataLoader]:
    """使用 `torch.utils.data.ImageFloder` 加载数据集, 返回三个 DataLoader (训练集, 验证集, 测试集)

    Args:
        `split (tuple[float])`: 训练集, 验证集, 测试集的占比\n
        `path (str, optional)`: 数据文件夹路径.\n
        `batchsize (int, optional)`: 批量大小. Defaults to 16.\n
        `transform (transforms.Compose, optional)`: 图像预处理函数. Defaults to transforms.ToTensor().\n
        `threads (int, optional)`: 加载数据集用的线程数. Defaults to cpu_count().\n

    Returns:
        `tuple[torchdata.DataLoader]`: 训练集, 验证集, 测试集的 DataLoader.\n
    """
    dataset: torchdata.Dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    train_indice = []
    val_indice = []
    test_indice = []
    labels = len(dataset.class_to_idx)

    for label in range(labels):
        label_indice = [dataset.imgs.index(i) for i in dataset.imgs if i[-1] == label]
        random.shuffle(label_indice)

        train_count = floor(len(label_indice) * split[0])
        val_count = floor(len(label_indice) * split[1])

        train_indice += label_indice[:train_count]
        val_indice += label_indice[train_count:train_count + val_count]
        test_indice += label_indice[train_count + val_count:]

    trainset = torchdata.Subset(dataset, train_indice)
    valset = torchdata.Subset(dataset, val_indice)
    testset = torchdata.Subset(dataset, test_indice)

    trainloader = torchdata.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=threads)
    valloader = torchdata.DataLoader(valset, batch_size=batchsize, shuffle=True, num_workers=threads)
    testloader = torchdata.DataLoader(testset, batch_size=batchsize, shuffle=True, num_workers=threads)
    return trainloader, valloader, testloader


def lmdbdataloader(split: tuple[float],
                   path: str,
                   batchsize: int = 16,
                   transform: transforms.Compose = transforms.ToTensor(),
                   threads: int = cpu_count()
                   ) -> tuple[torchdata.DataLoader]:
    """使用 `torch.utils.data.ImageFloder` 加载数据集, 返回三个 DataLoader (训练集, 验证集, 测试集)

    Args:
        `split (tuple[float])`: 训练集, 验证集, 测试集的占比\n
        `path (str, optional)`: 数据文件夹路径.\n
        `batchsize (int, optional)`: 批量大小. Defaults to 16.\n
        `transform (transforms.Compose, optional)`: 图像预处理函数. Defaults to transforms.ToTensor().\n
        `threads (int, optional)`: 加载数据集用的线程数. Defaults to cpu_count().\n

    Returns:
        `tuple[torchdata.DataLoader]`: 训练集, 验证集, 测试集的 DataLoader.\n
    """
    dataset: torchdata.Dataset = ImageFolderLMDB(path, transform=transform)

    trainloader = torchdata.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=threads)
    valloader = torchdata.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=threads)
    testloader = torchdata.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=threads)
    return trainloader, valloader, testloader
