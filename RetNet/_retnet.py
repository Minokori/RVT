
import torch
import torch.nn as nn

from ._config import *
from ._embeddings import *
from ._retention import MultiScaleRetention
from ._sequence import Image2Seq, Seq2Img2Seq, Seq2Prob


class RetNet(nn.Module):
    def __init__(self, layers: int, dim: int, ffn_size: int, heads: int, double_v_dim: bool = False):
        """RetNet\n
        ---\n
        >>> forward(X)  # (b, l, d) --> (b, l, d)
            forward_recurrent(x)  # (b, 1, d), [m][n](b, d/n, V/n) --> (b, 1, d), [m][n](b, d/n, V/n)
            forward_chunk(x)  # (b, c, d), [m][n](b, d/n, V/n) --> (b, c, d), [m][n](b, d/n, V/n)
        Args:\n
            `layers (int)`: MSR 层数\n
            `dim (int)`: 序列隐藏特征维度, 简记为 d\n
            `ffn_size (int)`: 前馈神经网络隐藏层大小\n
            `heads (int)`: MSR头的个数, 简记为 n\n
            `double_v_dim (bool, optional)`: 扩充 Value投影特征维度标志位, 简记为 `F`. Defaults to False.\n
        """
        super(RetNet, self).__init__()
        self.layers = layers
        """MSR 层数, 简记为 m"""
        self.dim = dim
        """序列隐藏特征维度, 简记为 d"""
        self.ffn_size = ffn_size
        """前馈神经网络隐藏层大小"""
        self.heads = heads
        """MSR头的个数, 简记为 n"""
        self.v_dim = dim * 2 if double_v_dim else dim
        """ 序列 x 的 V 投影的 维度, 简记为 `V`\n
        ---
        >>> V = (1 + F) * d = v * n
        """
        self.retentions: list[MultiScaleRetention] = nn.ModuleList([
            MultiScaleRetention(dim, heads, double_v_dim)
            for _ in range(layers)
        ])
        """Retention 块的 List
        >>> # (b, l, d) --> (b, l, d)
        """
        self.ffns: list[nn.Module] = nn.ModuleList([
            nn.Sequential
            (
                nn.Linear(dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, dim),
                # TEST dropout
                nn.Dropout(0.5),
            )
            for _ in range(layers)
        ])
        """前馈神经网络模块的 List
        >>> # (..., d) --> (..., d)
        """
        self.layer_norms_1: list[nn.LayerNorm] = nn.ModuleList([
            nn.LayerNorm(dim)
            for _ in range(layers)
        ])
        """LayerNorm模块的 List
        >>> # (..., d) --> (..., d)
        """
        self.layer_norms_2: list[nn.LayerNorm] = nn.ModuleList([
            nn.LayerNorm(dim)
            for _ in range(layers)
        ])
        """LayerNorm模块的 List
        >>> # (..., d) --> (..., d)
        """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """并行推理  前向传播\n
        >>> # (b, l, d) --> (b, l, d)\n
        ---\n
        Args:\n
            `X (torch.Tensor)`: 序列, shape = (b, l, d)\n
        Returns:
            `torch.Tensor`: shape = (b, l, d)\n
        """
        for i in range(self.layers):
            Y = (self.retentions[i](self.layer_norms_1[i](X)) + X)
            # shape = (b, l, d)
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y
            # shape = (b, l, d)
        return X

    def forward_recurrent(self, x_n: torch.Tensor, s_n_1s: list[list[torch.Tensor]], n: int) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        """循环表示  前向传播\n
        >>> # (b, 1, d) --> (b, 1, d), [m][n](b, d/n, V/n)\n
        ---\n
        Args:\n
            `x_n (torch.Tensor)`: 序列的第 i 个元素, shape = (b, 1, d)\n
            `s_n_1s (list[list[torch.Tensor]])`: 所有层的MSR头的状态列表, shape = [m][n](b, d/n, V/n)\n
            `n (int)`: 时间步|元素 索引\n
        Returns:\n
            `tuple[torch.Tensor, list[list[torch.Tensor]]]`: output[i] 和 state[i]\n
                output[i].shape = (b, 1, d)\n
                state[i].shape = [m][n](b, d/n, V/n)
        """
        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            # x_n.shape = (b, 1, d)     s_n_1s[i].shape = [n](b, d/n, V/n)
            # o_n.shape = (b, 1, d)     s_n.shape = [n](b, d/n, V/n)
            y_n = o_n + x_n
            # shape = (b, 1, d)
            s_ns.append(s_n)
            # shape = [m][n](b, d/n, V/n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n
            # shape = (b, 1, d)
        return x_n, s_ns  # shape = (b, 1, d) | [m][n](b, d/n, V/n)

    def forward_chunkwise(self, x_i: torch.Tensor, r_i_1s: list[list[torch.Tensor]], i: int) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        """分块循环  前向传播\n
        >>> #  (b, c, d) --> (b, c, d) | [m][n](b, d/n, V/n)\n
        ---\n
        Args:\n
            `x_i (torch.Tensor)`: 序列的第 i 个 chunk, shape = (b, c, d)\n
            `r_i_1s (list[list[torch.Tensor]])`: 所有层的MSR头的状态列表, shape = [m][n](b, d/n, V/n)\n
            `i (int)`: 时间步|块 索引\n\n
        Returns:
            `tuple[torch.Tensor, list[list[torch.Tensor]]]`: output[i] 和 state[i]\n
                output[i].shape = (b, c, d)\n
                state[i].shape = [m][n](b, d/n, V/n)
        """
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            # x_i.shape = (b, c, d) r_i_1s[j].shape = [n](b, d/n, V/n)
            # o_i.shape = (b, d/n, d)
            # r_i.shape = [n](b, d/n, V/n)
            y_i = o_i + x_i
            # shape = (b, c, d)
            r_is.append(r_i)
            # shape = [m][n](b, d/n, V/n)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i
            # shape = (b, c, d)
        return x_i, r_is  # shape = (b, c, d) | [m][n](b, d/n, V/n)


class ImageRetNet(nn.Module):
    """一个适用于图像分类的端到端 RetNet 网络\n
        + 继承自 `torch.nn.Module`
    """

    def __init__(self,
                 imgcfg: ImgInputConfig,
                 netcfg: RetNetConfig,
                 clscfg: LabelOutputConfig) -> None:
        """一个适用于图像分类的端到端 RetNet 网络\n
        ---\n
        >>> # (b, c, h, w) --> (b, cls)

        Args:
            `imgcfg (ImgInputConfig)`: 图像配置\n
            `netcfg (RetNetConfig)`: 网络配置\n
            `clscfg (LabelOutputConfig)`: 分类标签配置\n
        """
        super().__init__()

        self.img2seq = Image2Seq(
            imgcfg.size,
            imgcfg.channel,
            netcfg.sequence_length,
            netcfg.dimension)
        """图像序列化网络块\n
        >>> # (b, c, h, w) --> (b, l, d)
        """
        self.retnet = RetNet(netcfg.layer,
                             netcfg.dimension,
                             netcfg.ffn_size,
                             netcfg.head,
                             netcfg.vflag)
        """RetNet网络块\n
        >>> # (b, l, d) --> (b, l, d)
        """
        self.seq2prob = Seq2Prob(netcfg.dimension,
                                 clscfg.num)
        """输出网络块\n
        >>> # (b, l, d) --> (b, cls)
        """

    def forward(self, x) -> torch.Tensor:
        x = self.img2seq(x)
        x = self.retnet(x)
        x = self.seq2prob(x)
        return x


class ImageRetResNet(nn.Module):

    def __init__(self,
                 channels=32,
                 downscale=4,
                 device=None) -> None:
        """一个适用于图像分类的端到端 RetNet 网络\n
        ---\n
        >>> # (b, c, h, w) --> (b, cls)

        Args:

        """
        super().__init__()
        self.chunknum = 16
        self.headnum = 4
        self.device = torch.device("cpu") if device is None else device
        self.dense_embedding = nn.ModuleList([
            DenseFeature(3, channels * 1, 10, downscale),
            DenseFeature(channels * 1),
            DenseFeature(channels * 2),
            DenseFeature(channels * 4)
        ])
        """
        + 第 0 层 使输入变为 (b, c, H/s, W/s)\n
        + 其他层 使通道*2, H/2,W/2
        """

        self.retention: list[MultiScaleRetention] = nn.ModuleList([
            MultiScaleRetention(channels * 1, self.headnum),

            MultiScaleRetention(channels * 2, self.headnum),

            MultiScaleRetention(channels * 4, self.headnum),
            MultiScaleRetention(channels * 4, self.headnum),

            MultiScaleRetention(channels * 8, self.headnum),
            MultiScaleRetention(channels * 8, self.headnum),
        ])
        # 编码
        self.xpos_embbeding = nn.ModuleList([
            XPOS(channels * 1),

            XPOS(channels * 2),

            XPOS(channels * 4),
            XPOS(channels * 4),

            XPOS(channels * 8),
            XPOS(channels * 8)
        ])

        self.seq2img2seq = nn.ModuleList([
            Seq2Img2Seq(channels * 1),
            Seq2Img2Seq(channels * 2),
            Seq2Img2Seq(channels * 4),
            Seq2Img2Seq(channels * 8),
        ])
        """序列转图像再转序列,长度/4,维度*2
        """

        self.seq2prob = Seq2Prob(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y1 = self._stage_1(x)
        x, y2 = self._stage_2(x, y1)
        # x, y3 = self._stage_3(x, y2)
        x, y3 = self._stage_no_chunk_3(x, y2)
        # y4 = self._stage_4(x, y3)
        y4 = self._stage_no_chunk_4(x, y3)
        y5 = self.seq2prob(y4)
        # x, y1 = self._stage_no_chunk_1(x)
        # x, y2 = self._stage_no_chunk_2(x, y1)

        # y5 = self.seq2prob(y4)
        return y5

    def _stage_1(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_embedding[0](x)  # (b, 3,256,256) --> (b, 32, 64, 64)
        x1 = x.flatten(2).transpose(1, 2)  # (b,4096,32)
        # region retention
        b, l, d = x1.shape
        y1 = []
        chunksize = l // self.chunknum
        _r = [torch.zeros(b, d // self.headnum, d // self.headnum).to(self.device) for j in range(self.headnum)]
        for i in range(self.chunknum):
            _y, _r = self.retention[0].forward_chunkwise(x1[:, i * chunksize:(i + 1) * chunksize, :], _r, i)
            y1.append(_y)
        # endregion
        y1 = torch.cat(y1, dim=1)  # (b, 4096, 32)
        y1 = self.seq2img2seq[0](y1 + x1)  # (b, 1024, 64)
        return x, y1

    def _stage_2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.dense_embedding[1](x)  # (b, 32, 64, 64) --> (b, 64, 32,32)
        x2 = x.flatten(2).transpose(1, 2)  # (b, 1024 ,64)
        # region retention
        b, l, d = x2.shape
        y2 = []
        chunksize = l // self.chunknum
        _r = [torch.zeros(b, d // self.headnum, d // self.headnum).to(self.device) for j in range(self.headnum)]
        for i in range(self.chunknum):
            _y, _r = self.retention[1].forward_chunkwise(y[:, i * chunksize:(i + 1) * chunksize, :], _r, i)
            y2.append(_y)
        # endregion
        y2 = torch.cat(y2, dim=1)  # (b, 1024, 64)
        y2 = self.seq2img2seq[1](y2 + x2)  # (b, 256, 128)
        return x, y2

    def _stage_3(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.dense_embedding[2](x)  # (b, 64, 32,32) --> (b, 128, 16, 16)
        x3 = x.flatten(2).transpose(1, 2)  # (b, 256 ,64)
        # region retention
        b, l, d = x3.shape
        y3 = []
        chunksize = l // self.chunknum
        _r = [torch.zeros(b, d // self.headnum, d // self.headnum).to(self.device) for j in range(self.headnum)]
        for i in range(self.chunknum):
            _y, _r = self.retention[2].forward_chunkwise(y[:, i * chunksize:(i + 1) * chunksize, :], _r, i)
            y3.append(_y)
        # endregion
        y3 = torch.cat(y3, dim=1)  # (b, 256, 64)
        # region retention2
        y4 = []
        _r = [torch.zeros(b, d // self.headnum, d // self.headnum).to(self.device) for j in range(self.headnum)]
        for i in range(self.chunknum):
            _y, _r = self.retention[3].forward_chunkwise(y3[:, i * chunksize:(i + 1) * chunksize, :], _r, i)
            y4.append(_y)
        # endregion
        y4 = torch.cat(y4, dim=1)  # (b, 256, 64)
        y4 = self.seq2img2seq[2](y4 + x3)  # (b, 64, 128)
        return x, y4

    def _stage_4(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.dense_embedding[3](x)  # (b, 128, 16, 16) --> (b, 256, 8, 8)
        x4 = x.flatten(2).transpose(1, 2)  # (b, 64, 256)
        # region retention
        b, l, d = x4.shape
        y4 = []
        chunksize = l // self.chunknum
        _r = [torch.zeros(b, d // self.headnum, d // self.headnum).to(self.device) for j in range(self.headnum)]
        # retention1
        for i in range(self.chunknum):
            _y, _r = self.retention[4].forward_chunkwise(y[:, i * chunksize:(i + 1) * chunksize, :], _r, i)
            y4.append(_y)
        # endregion
        y4 = torch.cat(y4, dim=1)  # (b, 64, 256)
        # region retention2
        y5 = []
        _r = [torch.zeros(b, d // self.headnum, d // self.headnum).to(self.device) for j in range(self.headnum)]
        for i in range(self.chunknum):
            _y, _r = self.retention[5].forward_chunkwise(y4[:, i * chunksize:(i + 1) * chunksize, :], _r, i)
            y5.append(_y)
        # endregion
        y5 = torch.cat(y5, dim=1)  # (b, 64, 256)

        y4 = self.seq2img2seq[3](y4 + x4)  # (b, 16, 512)
        return y4

    def _stage_no_chunk_1(self, x: torch.Tensor):
        x = self.dense_embedding[0](x)  # (b, 3,256,256) --> (b, 32, 64, 64)
        _x = x.flatten(2).transpose(1, 2)  # (b,4096,32)
        y = self.retention[0].forward(_x)  # (b,4096,32)
        y = self.seq2img2seq[0](y + _x)  # (b,4096,32)
        return x, y

    def _stage_no_chunk_2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.dense_embedding[1](x)  # (b, 32, 64, 64) --> (b, 64, 32,32)
        _x = x.flatten(2).transpose(1, 2)  # (b, 1024 ,64)
        y = self.retention[1].forward(_x)
        y = self.seq2img2seq[1](y + _x)
        return x, y

    def _stage_no_chunk_3(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.dense_embedding[2](x)
        _x = x.flatten(2).transpose(1, 2)
        y = self.retention[2].forward(_x)
        y = self.retention[3].forward(y)
        y = self.seq2img2seq[2](_x + y)
        return x, y

    def _stage_no_chunk_4(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.dense_embedding[3](x)
        _x = x.flatten(2).transpose(1, 2)
        y = self.retention[4].forward(_x)
        y = self.retention[5].forward(y)
        y = self.seq2img2seq[3](_x + y)
        return y
