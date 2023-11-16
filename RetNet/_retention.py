"""Retention 块\n
---\n
Usage:
    >>> from retention import MultiScaleRetention as MSR
"""

import math
import torch
import torch.nn as nn
from typing import Union
from ._embeddings import XPOS

class SimpleRetention(nn.Module):
    def __init__(self, dim: int, gamma: float, size: int = None, double_v_dim: bool = False):
        """Simple retention mechanism based on the paper:\n
        "Retentive Network: A Successor to Transformer for Large Language Models"\n
        https://arxiv.org/pdf/2307.08621.pdf\n
        >>> # (b, l, d) --> (b, l, v)\n
        ---\n
        Args:\n
            `dim (int)`: 序列隐藏特征维度, 简记为 d \n
            `gamma (float)`: D 矩阵 使用的 γ 值\n
            `size (int, optional)`: 单个MSR头的特征维度, 简记为 d/n.\n
                n 为 MSR头的个数\n
                Defaults to d.\n
            `double_v_dim (bool, optional)`: 扩充 Value投影特征维度标志位, 简记为 F. Defaults to False.\n
        """
        super(SimpleRetention, self).__init__()

        self.dim = dim
        """隐藏特征维度, 简记为 `d`
        """
        # if size is None:
        #     size = dim
        self.size = size or dim
        """单个MSR头的特征维度, 简记为 `d/n`\n
            `n` 为 MSR头的个数\n
        """

        self.v_dim = size * 2 if double_v_dim else size
        """ 序列 x 的 V 投影的 维度, 简记为 `v`\n
        ---
        >>> v = (1 + F) * d/n = V/n
            V = (1 + F) * d
        """
        self.gamma = gamma
        """生成 D矩阵 使用的 γ 值
        """

        self.W_Q = nn.Parameter(torch.randn(dim, size) / dim)
        """将 序列x 映射为 Query 投影的权重矩阵
        + 可学习
        + 初始化符合正态分布 N ~ (0, 1/d)
        + shape = (d, d/n)
        """
        self.W_K = nn.Parameter(torch.randn(dim, size) / dim)
        """将 序列x 映射为 Key 投影的权重矩阵
        + 可学习
        + 初始化符合正态分布 N ~ (0, 1/d)
        + shape = (d, d/n)
        """
        self.W_V = nn.Parameter(torch.randn(dim, self.v_dim) / dim)
        """将 序列x 映射为 Value 投影的权重矩阵
        + 可学习
        + 初始化符合正态分布 N ~ (0, 1/d)
        + shape = (d, v)
        """
        self.xpos: nn.Module = XPOS(size)
        """对 序列x 进行 `XPOS` 编码的 编码层\n
        >>> # (b, l, d/n) --> (b, l, d/n)
        """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Parallel (default) representation of the retention mechanism.\n
        并行训练  前向传播\n
        >>> # (b, l, d) --> (b, l, v)\n
        ---\n
        Args:
            `X (torch.Tensor)`: shape = (b, l, d)\n
        Returns:
            `torch.Tensor`: shape = (b, l, v)\n
        """
        # X.shape = (b, l, d)

        sequence_length = X.shape[1]
        D: torch.Tensor = self._get_D(sequence_length).to(X)

        # shape = (l, l)
        Q: torch.Tensor = (X @ self.W_Q)
        # shape = (b, l, d/n)
        K: torch.Tensor = (X @ self.W_K)
        # shape = (b, l, d/n)
        Q = self.xpos(Q)
        # shape = (b, l, d/n)
        K = self.xpos(K, downscale=True)
        # shape = (b, l, d/n)
        V = X @ self.W_V
        # shape = (b, l, v)
        ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)
        # tensor1.shape = (b, l, l)
        # tensor2.shape = (1, l, l)
        # shape = (b, l, l)
        # TEST dropout
        return (nn.Dropout(0.5)(ret)) @ V  # shape = (b, l, v)

    def forward_recurrent(self, x_n: torch.Tensor, s_n_1: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent representation of the retention mechanism.\n
        循环推理  前向传播\n
        >>> # (b, 1, d) --> (b, 1, v), (b, d/n, v)\n
        ---\n
        Args:
            `x_n (torch.Tensor)`: 序列的第 n 个元素 shape = (b, 1, d)\n
            `s_n_1 (torch.Tensor)`: n-1 步的隐藏状态 shape = (b, d/n, v)\n
            `n (int)`: 时间步|元素 索引\n

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: output[n] 和 state[n]\n
                output.shape = (b, 1, v)\n
                state.shape = (b, d/n, v)
        """

        Q = (x_n @ self.W_Q)
        # shape = (b, 1, d/n)
        K = (x_n @ self.W_K)
        # shape = (b, 1, d/n)

        Q = self.xpos(Q, offset=n + 1)
        # shape = (b, 1, d/n)
        K: torch.Tensor = self.xpos(K, offset=n + 1, downscale=True)
        # shape = (b, 1, d/n)

        V = x_n @ self.W_V
        # shape = (b, 1, v)

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)
        # shape = (b, d/n, v)

        return (Q @ s_n), s_n  # shape = (b, 1, v), (b, d/n, v)

    def forward_chunkwise(self, x_i: torch.Tensor, r_i_1: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunkwise representation of the retention mechanism.\n
        分块循环  前向传播\n
        >>> # (b, c, d) --> (b, c, v)\n
        ---\n
        Args:
            `x_i (torch.Tensor)`: 序列的第 i 个块, shape = (b, c, d)\n
            `r_i_1 (torch.Tensor)`: 第 i-1 步的隐藏状态, shape = (b, d/n, v)\n
            `i (int)`: 时间步|元素 索引\n
        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: output[n] 和 state[n]\n
                output.shape = (b, c, v)\n
                state.shape = (b, d/n, v)\n
        """
        # x_i.shape = (b, c, d)
        # r_i_1.shape = (b, d/n, v)
        batch, chunk_size, _ = x_i.shape
        # TODO 不好看
        D = self._get_D(chunk_size).to(r_i_1.device)
        # shape = (c, c)
        Q = (x_i @ self.W_Q)
        # shape = (b, c, d/n)
        K: torch.Tensor = (x_i @ self.W_K)
        # shape = (b, c, d/n)
        Q = self.xpos(Q, offset=i * chunk_size)
        # shape = (b, c, d/n)
        K = self.xpos(K, offset=i * chunk_size, downscale=True)

        # shape = (b, c, d/n)
        V = x_i @ self.W_V
        # shape = (b, c, v)

        r_i = (K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1
        # shape = (b, d/n, v)
        # NOTE
        # region
        # shape:
        # K.transpose --> (b, d/n, c)
        # D[-1] --> (c) | .view --> (1, c, 1)
        # ---
        # (b, d/n,c) @ ((b, c, v) * (1, c, 1)) --> (b, d/n,c) @ (b, c, v)
        # endregion
        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V
        # shape = (b, c, v)
        # TODO 不好看
        e = torch.zeros(batch, chunk_size, 1).to(r_i_1.device)
        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)
        # shape = (b, c, 1)
        # e[b, i ,0] = gamma ** (i + 1)
        cross_chunk = (Q @ r_i_1) * e
        # shape = (b, c, v)
        return inner_chunk + cross_chunk, r_i  # shape = (b, c, v), (b, d/n, v)

    def _get_D(self, sequence_length: int) -> torch.Tensor:
        """生成一个 `shape = (sequence_length, sequence_length)` 的下三角阵 D

        Args:
            `sequence_length (int)`: 矩阵大小\n

        Returns:
            `torch.Tensor`: D 矩阵\n
        """
        n = torch.arange(sequence_length).unsqueeze(1)
        # shape = (seq_len, 1)
        m = torch.arange(sequence_length).unsqueeze(0)
        # shape = (1, seq_len)
        D = (self.gamma ** (n - m)) * (n >= m).float()
        # (n - m).shape = (n >=m).shape=(seq_len, seq_len)
        D[D != D] = 0
        return D


class MultiScaleRetention(nn.Module):
    def __init__(self, dim: int, heads: int, double_v_dim=False):
        """Multi-scale retention mechanism based on the paper:\n
        "Retentive Network: A Successor to Transformer for Large Language Models"\n
        https://arxiv.org/pdf/2307.08621.pdf\n
        >>> # (b, l, d) --> (b, l, d)\n
        ---\n
        Args:
            `dim (int)`: 序列隐藏特征维度, 简记为 d\n
            `heads (int)`: MSR头的个数, 简记为 n\n
            `double_v_dim (bool, optional)`: 扩充 Value投影特征维度标志位, 简记为 `F`. Defaults to False.\n
        """
        super(MultiScaleRetention, self).__init__()
        self.dim = dim
        """隐藏特征维度, 简记为 `d`"""
        self.v_dim = dim * 2 if double_v_dim else dim
        """ 序列 x 的 V 投影的 维度, 简记为 `V`\n
        ---
        >>> V = (1 + F) * d = v * n
        """
        self.heads = heads
        """MSR 头的个数, 简记为 n\n"""
        assert dim % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = dim // heads
        """单个MSR头的特征维度, 简记为 `d/n`\n
        """
        self.gammas: list[float] = (1 - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach().cpu().tolist()
        """各 MSR 头用的gamma值\n
        + len = n"""
        self.swish: nn.Module = lambda x: x * torch.sigmoid(x)
        """swish 激活函数
        + output.shape = input.shape"""
        self.W_G = nn.Parameter(torch.randn(dim, self.v_dim) / dim)
        """MSR 的 W_G 权重矩阵
        + 可学习
        + 初始化符合正态分布 N ~ (0, 1/d)
        + shape = (d, V)
        """
        self.W_O = nn.Parameter(torch.randn(self.v_dim, dim) / dim)
        """MSR 的 W_G 权重矩阵
        + 可学习
        + 初始化符合正态分布 N ~ (0, 1/d)
        + shape = (V, d)
        """
        self.group_norm = nn.GroupNorm(heads, self.v_dim)
        """将 tensor 的 第一个维度分为 n 组, 每组进行标准化\n
        + tensor 的第二个维度必须为 V
        """
        self.retentions: list[SimpleRetention] = nn.ModuleList([
            SimpleRetention(self.dim, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])
        """Retention 块的 List
        >>> # (b, l, d) --> (b, l, V/n)
        """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        parallel representation of the multi-scale retention mechanism\n
        并行训练  前向传播\n
        >>> # (b, l, d) --> (b, l, d)\n
        ---\n
        Args:
            `X (torch.Tensor)`: shape = (b, l, d)\n
        Returns:
            `torch.Tensor`: shape = (b, l, d)\n
        """

        # apply each individual retention mechanism to X
        Y: list[torch.Tensor] = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X))
        # X.shape = (b, l, d)
        # Y.item.shape = (b, l, V/n)

        Y: torch.Tensor = torch.cat(Y, dim=2)
        # shape = (b, l, V/n*n) = (b, l, V)
        Y_shape = Y.shape
        Y: torch.Tensor = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        # region
        # 1. reshape --> (b*l, V)
        # 2. reshape --> (b, l, V)
        # endregion
        # shape = (b, l, V)
        # TEST dropout
        return nn.Dropout(0.5)((self.swish(X @ self.W_G) * Y) @ self.W_O)

    def forward_recurrent(self, x_n: torch.Tensor, s_n_1s: list[torch.Tensor], n: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """recurrent representation of the multi-scale retention mechanism\n
        循环推理  前向传播\n
        >>> (b, 1, d) --> (b, 1, d), [n](b, d/n, V/n)
        ---\n
        Args:\n
            `x_n (torch.Tensor)`: 序列的第 n 个元素 shape = (b, 1, d)\n
            `s_n_1s (list[torch.Tensor])`: 所有 MSR 头的 n-1 步的隐藏状态, shape = [n](b, d/n, V/n)\n
            `n (int)`: 时间步|元素 索引\n

        Returns:
            `tuple[torch.Tensor, list[torch.Tensor]]`: output[n] 和 state[n]\n
                output.shape = (b, 1, d)\n
                state.shape = [n](b, d/n, V/n)
        """
        # apply each individual retention mechanism to a slice of X
        Y: list[torch.Tensor] = []
        s_ns: list[torch.Tensor] = []
        for i in range(self.heads):  # i 0->n-1
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, :, :],  # x_n[:,:,:].shape = (b, 1, d)
                s_n_1s[i],  # s_n_1s[i].shape = (b, d/n, V/n)
                n
            )
            # y.shape = (b, 1, V/n)
            # s_n.shape = (b, d/n, V/n)
            Y.append(y)
            s_ns.append(s_n)
        Y: torch.Tensor = torch.cat(Y, dim=2)

        # shape = (b, 1, V/n*n) = (b, 1, V)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        # shape = (b, 1, V/n*n) = (b, 1, V)
        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns  # shape = (b, 1, d), [n](b, d/n, V/n)

    def forward_chunkwise(self, x_i: torch.Tensor, r_i_1s: Union[list[torch.Tensor], torch.Tensor], i: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Chunkwise representation of the retention mechanism.\n
        分块循环  前向传播\n
        >>> # (b, c, d) --> (b, c, V), [n](b, d/n, V/n)\n
        ---\n
        Args:\n
            `x_i (torch.Tensor)`: 序列的第 i 个 chunk, shape = (b, c, d)\n
            `r_i_1s (list[torch.Tensor])`: 各 MSR 头的第 i-1 步的隐藏状态, shape = [n](b, d/n, V/n)\n
            `i (int)`: 时间步|块 索引\n
        Returns:\n
            `tuple[torch.Tensor, list[torch.Tensor]]`: output[n] 和 state[n]\n
                output.shape = (b, c, V)\n
                state.shape = [n](b, d/n, V/n)\n
        """
        # apply each individual retention mechanism to a slice of X
        Y: list[torch.Tensor] = []
        # TEST
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(
                x_i[:, :, :],  # shape = (b, c, d)
                r_i_1s[j],  # r_i_1s.shape = [n](b, d/n, V/n)
                i
            )
            # y.shape = (b, c, V/n)
            # r_i.shape = (b, d/n, V/n)
            Y.append(y)
            # Y.shape = [n](b, c, V/n)
            r_is.append(r_i)
            # r_is.shape = [n](b, d/n, V/n)
        # if isinstance(r_i_1s, list):
        #     r_is = []
        #     for j in range(self.heads):
        #         y, r_i = self.retentions[j].forward_chunkwise(
        #             x_i[:, :, :],  # shape = (b, c, d)
        #             r_i_1s[j],  # r_i_1s.shape = [n](b, d/n, V/n)
        #             i
        #         )
        #         # y.shape = (b, c, V/n)
        #         # r_i.shape = (b, d/n, V/n)
        #         Y.append(y)
        #         # Y.shape = [n](b, c, V/n)
        #         r_is.append(r_i)
        #         # r_is.shape = [n](b, d/n, V/n)

        # elif isinstance(r_i_1s, torch.Tensor):
        #     r_is = torch.Tensor(device=r_i_1s.device)
        #     for j in range(self.heads):
        #         y, r_i = self.retentions[j].forward_chunkwise(
        #             x_i[:, :, :],
        #             r_i_1s[j, :, :, :],  # r_i_1s.shape = (n, b, d/n, V/n),
        #             i
        #         )

        Y: torch.Tensor = torch.cat(Y, dim=2)
        # Y.shape = (b, c, V)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        # Y.shape = (b, c, V)

        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is  # shape = (b, c, V) | [n](b, d/n, V/n)
