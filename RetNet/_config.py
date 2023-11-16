from dataclasses import dataclass
from math import floor, sqrt, log2
import warnings


@dataclass
class ImgInputConfig:
    """图片输入配置"""
    size: int
    """图片边长, 简记为 `h`, `w`"""
    channel: int = 3
    """图片通道, 简记为 `c`"""


@dataclass
class RetNetConfig:
    """RetNet 配置"""
    sequence_length: int
    """序列长度, 简记为 `l`"""
    layer: int
    """MSR 层数, 简记为 `m`"""
    dimension: int
    """序列隐藏层维度, 简记为 `d`"""
    ffn_size: int
    """前馈神经网络隐藏层大小"""
    head: int
    vflag: bool = False

    def __post_init__(self):
        if sqrt(self.sequence_length) % 1 != 0:
            raise ValueError("序列长度的平方根必须为整数")
        if log2(self.ffn_size) % 1 != 0:
            warnings.warn(f"前馈神经网络隐藏层大小: {self.ffn_size}, 推荐为2的次方, 考虑: {2 ** floor(log2(self.ffn_size))}", UserWarning)


@dataclass
class LabelOutputConfig:
    num: int
