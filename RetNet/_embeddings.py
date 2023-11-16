"""XPOS 编码\n
---\n
Usage:
    >>> from xpos_relative_position import XPOS
        emmbeded_sequence = XPOS(sequence)\n
+ Copyright (c) 2022 Microsoft
+ Licensed under The MIT License (https://github.com/microsoft/torchscale/blob/main/LICENSE)
"""

import torch
import torch.nn as nn


def fixed_pos_embedding(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """序列 x 的固定编码矩阵\n
    >>> # shape: (l, d) --> (l, d), (l, d)\n
    ---\n
    Args:
        `x (torch.Tensor)`: shape = (l, d)\n
    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: 和x形状相同的两个编码矩阵\n
            tensor_sin.shape = (l, d)\n
            tensor_cos.shape = (l, d)
    """
    seq_len, dim = x.shape

    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))

    sinusoid_inp: torch.Tensor = (
        torch.einsum("i , j -> i j",
                     torch.arange(0, seq_len, dtype=torch.float),
                     inv_freq)
        .to(x)
        # NOTE
        # region
        # torch.einsum("i,j-> i j", input1, input2)
        # 将 input1[i] 和 input2[j] 的元素相乘, 放在 output[i][j]
        # 对于上面两个一维向量, 相当于:
        # output = input1.unsqueeze(-1) @ input2.unsqueeze(0)
        # output.shape = (input1.Size(), input2.Size())
        # ---
        # tensor.to(other)
        # 将 tensor 的 dytpe, device 和 other 保持一致
        # endregion
    )
    # sinusoid_inp.shape = (seq_len, dim)

    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """将序列 x 的最后两个维度进行旋转\n
    >>> # shape: (b, l, d) --> (b, l, d)\n
    ---\n
    Args:
        `x (torch.Tensor)`: shape = (b, l, d)\n
    Returns:
        `torch.Tensor`: 旋转后的张量\n
            output.shape = x.shape = (b, l, d)
    """
    x1 = x[:, :, ::2]
    # x1.shape = (b, seq_len, dim/2)

    x2 = x[:, :, 1::2]
    # x2.shape = (b, seq_len, dim/2)

    x = torch.stack((-x2, x1), dim=-1)
    # x.shape = (b, seq_len, dim/2, 2)

    return x.flatten(-2)  # shape = (b, seq_len, dim)


def duplicate_interleave(x: torch.Tensor) -> torch.Tensor:
    """将 x 的数据`按元素`重复, 并 reshape 为二维矩阵\n
    >>> # (d0, d1, ..., dn)--> (d0, 2 * d1*d2*...dn)\n
    ---\n
    Args:\n
        `x (torch.Tensor)`: shape = (d0, d2, ..., dn)\n
    Returns:
        `torch.Tensor`: shape = (d0, 2 * d1*d2*...*dn)\n
    """
    dim0 = x.shape[0]
    x = x.view(-1, 1)  # flatten the matrix
    # x.shape = (all_dim_mul, 1)
    x = x.repeat(1, 2)
    # x.shape = (all_dim_mul/2, 2)
    # NOTE
    # Tensor.repeat(dim0,dim1,...dimn)
    # 将 Tensor的 dim{x} 重复 dim{x} 次
    x = x.view(dim0, -1)
    # m.shape = (dim0, all_dim_mul*2/dim0)
    return x


def apply_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, scale=1.0) -> torch.Tensor:
    """向序列 x 应用旋转位置编码\n
    >>> # (b, l, d) --> (b, l, d)\n
    ---\n
    Args:
        `x (torch.Tensor)`: shape = (b, l, d)\n
        `sin (torch.Tensor)`: 位置编码奇数部分, shape = (l, d/2)\n
        `cos (torch.Tensor)`: 位置编码偶数部分, shape = (l, d/2)\n
        `scale (float, optional)`: 缩放大小. Defaults to 1.0\n

    Returns:
        `torch.Tensor`: shape = (b, l, d)\n
    """
    # x.shape = # x.shape = (b, l, d)
    # other.shape = (l, d/2)
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # shape = (l, d)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    """
    XPOS 编码\n
    >>> output.shape = input.shape = (batch size, seq_len, dim)
        # 简记为 (b, l, d)
    """

    def __init__(self, dimentions: int, scale_base=512):
        """XPOS 编码\n
        >>> # (b, l, d) --> (b, l, d)\n
        ---\n
        Args:
            `dimentions (int)`: 输入序列的特征维度, 简记为 `d`\n
            `scale_base (int, optional)`: 降缩放因子. Defaults to 512.\n
        """
        super().__init__()
        self.dimentions = dimentions
        """输入序列的特征维度, 简记为 `d`"""
        self.scale_base = scale_base
        """缩放因子"""
        self.register_buffer("scale", (torch.arange(0, dimentions, 2) + 0.4 * dimentions) / (1.4 * dimentions))
        # self.scale.shape = (d/2)

    def forward(self, x: torch.Tensor, offset=0, downscale=False) -> torch.Tensor:
        """XPOS  前向传播\n
        >>> # (b, l, d) --> (b, l, d)\n
        Args:
            `x (torch.Tensor)`: shape = (b, l, d)\n
            `offset (int, optional)`: 位置编码偏移量, 简记为 `Δ`. Defaults to 0.\n
            `downscale (bool, optional)`: 降缩放标志位. Defaults to False.\n

        Returns:
            `torch.Tensor`: shape = (b, l, d)\n
        """
        length = x.shape[1]
        min_pos = 0
        max_pos = length + offset + min_pos
        scale: torch.Tensor = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        # NOTE
        # region
        # x1.shape = (   d/2)
        # x2.shape = (l+Δ, 1)
        # 广播机制
        # 1. 最后一个维度一个存在一个为 1
        # 2. 倒数第二个一个存在一个不存在
        # 可以广播
        # x1.shape = (l+Δ, d/2)
        # x1.shape = (l+Δ, d/2)
        # endregion
        # shape = (l+Δ, d/2)
        sin, cos = fixed_pos_embedding(scale)
        # shape = (l+Δ, d/2)

        if scale.shape[0] > length:  # 即 offset != 0, 裁剪到和 length一样长
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        # shape = (l, d/2)

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        # x.shape = (b, l, d)
        return x

    def forward_reverse(self, x: torch.Tensor, offset=0, downscale=False):
        """类似 `self.forward()`

        >>> forward().scale = [0,..,2n)
        forward_reverse.scale = [-n,n)

        """
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale: torch.Tensor = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        # TEST dropout1
        return nn.Dropout()(x)


class DenseFeature(nn.Module):
    """卷积特征提取.
    """

    def __init__(self,
                 input: int,
                 output: int = None,
                 hidden: int = None,
                 downscale: int = 2,
                 ) -> None:
        """通过一个简单的稠密连接的卷积网络提取特征\n
        ---\n
        >>>   #(b, c, h, w) --> (b,  h/s*w/s, o)

        Args:
            `input (int)`: 输入图像的通道数, 简记为 d\n
            `output (int, optional)`: 输出图像的通道数, 简记为 o. Defaults to None.\n
                + 若 `output` 未指定, 则: `o = d * 2`\n
            `hidden (int, optional)`: Dense block 中的 growrate, 简记为 i. Defaults to None.\n
                + 若 `hidden` 未指定, 则: `i = d`\n
            `downscale (int, optional)`: 图像缩小比例, 简记为 s. Defaults to 2.\n

        """
        super().__init__()

        output = output or input * 2
        hidden = hidden or input
        self.featureconv = nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=hidden, kernel_size=1, stride=1),
            nn.BatchNorm2d(hidden),
            nn.GELU())
        """1 × 1卷积层\n
        >>>  # (*, *, h, w) --> (*, *, h, w)
        """

        self.avgpool = nn.AvgPool2d(downscale, downscale)
        """池化层.\n
        >>>  # (*, *, h, w) --> (*, *, h/s, w/s)
        """
        self.patch_conv = nn.Sequential(
            nn.Conv2d(hidden + input, output, kernel_size=downscale, stride=downscale),
            nn.BatchNorm2d(output),
            nn.GELU())
        """patch 层.\n
        + 用于扩充维度, 缩小图像大小
        >>>  # (d + i, *, h, w) --> (*, *, h/s, w/s)
        """

        self.conv3 = nn.Sequential(
            nn.Conv2d(input + hidden + output, output, 1, 1),
            nn.BatchNorm2d(output),
            nn.GELU())
        """`DenseNet` 中的 transition 层.\n
        + 用于扩充维度
        >>>  # (*, *, h, w) --> (*, *, h/s, w/s)
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y11 = self.featureconv(x)  # (b, i, h, w)
        y21 = self.patch_conv(torch.cat((y11, x), 1))  # (b, o, h/s, w/s)
        y22 = self.avgpool(y11)  # (b, i, h/s, w/s)
        y23 = self.avgpool(x)  # (b, i, h/s, w/s)
        y3 = torch.cat((y21, y22, y23), 1)  # (b, d+i+o, h/s, w/s)
        y4: torch.Tensor = self.conv3(y3)  # (b, o, h/s, w/s)
        return y4
