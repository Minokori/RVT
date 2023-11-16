from math import sqrt
import torch
from torch import nn


class Image2Seq(nn.Module):
    """
    将图片通过二维卷积层编码为序列。
    >>> # (b, c, h, w) --> (b, l, d)\n

    """

    def __init__(self, image_size: int, channel: int, seq_len: int, dimmension: int):
        """将图片通过二维卷积层编码为序列。\n
        >>> # (b, c, h, w) --> (b, l, d)\n
        ---\n
        Args:
            `image_size (int)`: 图片大小, 简记为 h,w\n
            `channel (int)`: 图片通道数, 简记为 c\n
            `seq_len (int)`: 序列长度, 简记为 l\n
            `dimmension (int)`: 序列隐藏特征维度, 简记为 d\n
        """
        super().__init__()
        self.image_size: int = image_size
        """图片大小, 简记为 `h`,`w`"""
        self.conv_size: tuple = self._compute_kernel_size(
            (image_size, image_size),
            (sqrt(seq_len), sqrt(seq_len))
        )
        """核大小\n
        ---
        >>> kernel_size = stride_size = conv_size
        """
        self.num_channels: int = channel
        """图片通道数, 简记为 `c`"""
        self.dim: int = dimmension
        """序列隐藏特征维度, 简记为 `d`"""
        self.num_patches: int = seq_len
        """序列长度, 简记为 `l`"""
        self.projection = nn.Conv2d(self.num_channels, self.dim, kernel_size=self.conv_size, stride=self.conv_size)
        """二维卷积层\n
        ---\n
        >>> # (b, c, h, w) --> (b, d, √l, √l)
        """
        self.apply(_init_params)

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        img = self.projection(img)
        # shape = (b, d, √l, √l)
        img = img.flatten(2)
        # shape = (b, d, l)
        img = img.transpose(1, 2)
        # shape = (b, l, d)
        return img

    @staticmethod
    def _compute_kernel_size(insize: tuple[int, int], outsize: tuple[int, int]) -> tuple[int, int]:
        conv_size = []
        for input_size, output_size in zip(insize, outsize):
            rem = (input_size) % output_size
            if rem != 0:
                raise ValueError(f"input size{insize} CANNOT be divided by expected output size{outsize}")
            else:
                conv_size.append(int(input_size // output_size))
        return tuple(conv_size)


class Seq2Prob(nn.Module):
    def __init__(self, dim: int, class_num: int):
        """从输出到分类\n
        ---\n
        >>> # (b, l, d) --> (b, cls)

        Args:
            `dim (int)`: 隐藏特征维度, 简记为 d\n
            `class_num (int)`: 类别标签个数, 简记为 cls\n
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, dim))
        self.linear = nn.Linear(dim, class_num)
        self.apply(_init_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TEST 相较于 ViT 新加的
        # x = nn.Dropout1d()(x)
        x = self.pool(x)
        x = x.squeeze(-2)
        x = self.linear(x)
        return x


def _init_params(module: nn.Module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)
        nn.init.normal_(module.bias)


class Seq2Img2Seq(nn.Module):

    def __init__(self, inc) -> None:
        """序列转图像再转序列,长度/4,维度*2

        Args:
            `inc (_type_)`: _description_\n
            `length (_type_)`: _description_\n
        """
        super().__init__()
        self.conv1 = nn.Conv2d(inc, inc * 2, 2, 2)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape  # (b, s*s, d)
        size = int(sqrt(L))
        x = x.transpose(1, 2)  # (b, d, s*s)
        x = x.view(B, D, size, size)  # (b, d, s,s)
        x = self.conv1(x)
        # (b,2d,s/2,s/2)
        x = x.flatten(2)  # (b,2d, s*s/4)
        x = x.transpose(1, 2)
        return nn.GELU()(x)  # (b,s*s/4,2d)
