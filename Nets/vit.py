import math
import torch
from torch import nn
import torch.nn.functional as F


class NewGELUActivation(nn.Module):
    """
    GELU 激活函数\n

    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    将图片通过二维卷积层编码为序列。

    >>> output.shape = (batch_size, num_patches, hidden_size)
    # batch_size 条 num_patch 长的序列, 序列有 hidden_size 个隐藏特征维度


    """

    def __init__(self, config: dict):
        """将图片通过二维卷积层编码为序列。

        >>> output.shape = (batch_size, num_patches, hidden_size)
        # batch_size 条 num_patch 长的序列, 序列有 hidden_size 个隐藏特征维度


        Args:
            `config (dict)`: 网络配置\n
        """
        super().__init__()
        self.image_size: int = config["image_size"]
        """图片大小(长和宽)"""
        self.patch_size = config["patch_size"]
        """裁剪块大小(长和宽)"""
        self.num_channels: int = config["num_channels"]
        """输入图片通道数"""
        self.hidden_size: int = config["hidden_size"]
        """隐藏特征维度"""

        # NOTE 图片为正方形，切片为正方形，切片数=(图片边长/切片边长)^2
        self.num_patches: int = (self.image_size // self.patch_size) ** 2
        """序列长度"""

        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        """将图像序列化的二维卷积层
        
            + 大小`patch_size×patch_size`, `num_channel`通道的图像切片\n
            + -->\n
            + 1×1像素点, 有hidden_size个隐藏特征(通道)
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        # NOTE x.shape = (batch_size, hidden_size, patch_size, patch_size)
        # batch_size条, hidden通道, patch_size×patch_size的图片
        x = x.flatten(2)
        # NOTE x.shape = (batch_size, hidden_size, num_patch)
        # batch_size条，hidden通道 num_patch长的序列
        x = x.transpose(1, 2)
        # NOTE x.shape = (batch_size, num_patch, hidden_size)
        # batch_size条 num_patch长的序列, 序列有 hidden_size 个隐藏特征维度
        return x


class Embeddings(nn.Module):
    """
    将图片编码为序列, 并添加标签token和位置编码。

    >>> output.shape = (batch_size, num_patch + 1, hidden_size)
    # batch_size条 num_patch+1 长的序列, 序列有 hidden_size 个隐藏特征维度

    """

    def __init__(self, config: dict):
        """将图片编码为序列, 并添加标签token和位置编码

        >>> output.shape = (batch_size, num_patch + 1, hidden_size)
        # batch_size条 num_patch+1 长的序列, 序列有 hidden_size 个隐藏特征维度

        Args:
            `config (dict)`: 网络配置\n
        """
        super().__init__()
        self.config = config
        """网络配置"""
        self.patch_embeddings = PatchEmbeddings(config)
        """图片序列化编码层"""

        self.cls_token: torch.Tensor = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        """序列类别标签编码"""
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        """序列位置编码"""
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        """dropout 层"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)
        # NOTE x.shape = (batch_size, num_patch, hidden_size)

        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # NOTE cls_tokens.shape = (batch_size, 1, hidden_size)
        x = torch.cat((cls_tokens, x), dim=1)
        # NOTE x.shape = (batch_size, num_patch+1, hidden_size)
        # 序列的第一个值(x[b][0][:])用于表示类别信息

        x = x + self.position_embeddings
        # REMIND 位置编码应该用 sin或者cos, 而非随机初始化
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    >>> attention_output, attention_probs = output
    attention_output.shape = (batch_size, sequence_length, attention_head_size)
    # attention_output[b][t][v]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第v个Value特征分量" 的注意力
    attention_probs.shape = (batch_size, sequence_length, sequence_length)
    # attention_probs[b][t][s]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性
    """

    def __init__(self, hidden_size: int, attention_head_size: int, dropout: float, bias=True):
        """单头自注意力机制

        >>> attention_output, attention_probs = output
        attention_output.shape = (batch_size, sequence_length, attention_head_size)
        # attention_output[b][t][v]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第v个Value特征分量" 的注意力
        attention_probs.shape = (batch_size, sequence_length, sequence_length)
        # attention_probs[b][t][s]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性

        Args:
            `hidden_size (int)`: 序列隐藏特征维度\n
            `attention_head_size (int)`: 注意力维度\n
            `dropout (float)`: dropout概率\n
            `bias (bool, optional)`: 偏置启用标志位. Defaults to True.\n
        """
        super().__init__()
        self.hidden_size = hidden_size
        """序列隐藏特征维度"""
        self.attention_head_size = attention_head_size
        """注意力特征维度"""
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        """生成`查询Query`的全连接层"""
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        """生成`键Key`的全连接层"""
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        """生成`值Value`的全连接层"""

        self.dropout = nn.Dropout(dropout)
        """dropout层"""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        # NOTE q|k|v.shape = (batch_size, sequence_length, attention_head_size)
        query: torch.Tensor = self.query(x)
        # Target的Query表示, query[b][i][q]: 第b组Target序列的索引i上的第q个Query特征分量
        key: torch.Tensor = self.key(x)
        # Source的Key表示, key[b][i][k]: 第b组Source序列的索引i上的第k个Key特征分量
        value: torch.Tensor = self.value(x)
        # Source的Value表示, value[b][i][v]: 第b组Source序列的索引i上的第v个Value特征分量

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # NOTE attention_scores.shape = (batch_size, sequence_length(T), sequence_length(S))
        # attention_scores[b][t][s]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性
        attention_probs = F.softmax(attention_scores, dim=-1)  # 将相关性用softmax缩放到0~1之间
        attention_probs = self.dropout(attention_probs)
        # NOTE attention_probs.shape = (batch_size, sequence_length(T), sequence_length(S))

        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # NOTE attention_output.shape = (batch_size, sequence_length(T), attention_head_size(S))
        # attention_output[b][t][v]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第v个Value特征分量" 的注意力
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """多头自注意力机制

    >>> attention_output, attention_probs = output
    attention_output.shape = (batch_size, sequence_length, hidden_size)
    # attention_output[b][t][h]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第h个隐藏特征分量" 的注意力
    attention_probs.shape = (batch_size, num_attention_heads, sequence_length(T), sequence_length(S))
    # attention_probs[b][i][t][s]: 第b组序列, 第i个单头, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性
    """

    def __init__(self, config: dict):
        """多头自注意力机制

        >>> attention_output, attention_probs = output
        attention_output.shape = (batch_size, sequence_length, hidden_size)
        # attention_output[b][t][h]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第h个隐藏特征分量" 的注意力
        attention_probs.shape = (batch_size, num_attention_heads, sequence_length(T), sequence_length(S))
        # attention_probs[b][i][t][s]: 第b组序列, 第i个单头, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性

        Args:
            `config (dict)`: 网络配置\n
        """
        super().__init__()
        self.hidden_size: int = config["hidden_size"]
        """隐藏层大小
        """
        self.num_attention_heads: int = config["num_attention_heads"]
        """多头自注意力机制的 `头数` """
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size: int = self.hidden_size // self.num_attention_heads
        """单头注意力机制的注意力特征维度"""
        self.all_head_size: int = self.num_attention_heads * self.attention_head_size
        """多头注意力机制的注意力特征维度"""
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias: bool = config["qkv_bias"]
        """偏置启用标志位"""
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        """注意力头"""
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        """将所有单头注意力维度变为隐藏层维度大小的全连接层"""
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
        """dropout 层"""

    def forward(self, x: torch.Tensor, output_attentions=False) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE x.shape = (batch_size, sequence_length, hidden_size)
        # x[b][i][h]: 第b组序列, 索引i, 第h维度的分量

        single_outputs: tuple(torch.Tensor, torch.Tensor) = [head(x) for head in self.heads]
        attention_output = torch.cat([attention_output for attention_output, _ in single_outputs], dim=-1)
        # NOTE attention_output.shape = (batch_size, seqence_length(T), attention_head_size(V))
        # torch.cat() --> (batch_size, seqence_length(T), all_head_size)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # NOTE --> (batch_size, seqence_length(T), hidden_size)
        # attention_output[b][t][h]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第h个隐藏特征分量" 的注意力
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_prob for _, attention_prob in single_outputs], dim=1)
            # NOTE attention_prob.shape = (batch_size, seqence_length(T),seqence_length(S))
            # NOTE attention_probs.shape = (batch_size, num_attention_heads, seqence_length(T),seqence_length(S))
            # attention_prob[b][i][t][s]: 第b组序列, 第i个单头, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """快速多头自注意力机制

    >>> attention_output, attention_probs = output
    attention_output.shape = (batch_size, sequence_length, hidden_size)
    # attention_output[b][t][h]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第h个隐藏特征分量" 的注意力
    attention_probs.shape = (batch_size, num_attention_heads, sequence_length(T), sequence_length(S))
    # attention_probs[b][i][t][s]: 第b组序列, 第n个注意力, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性
    """

    def __init__(self, config: dict):
        """快速多头自注意力机制

        >>> attention_output, attention_probs = output
        attention_output.shape = (batch_size, sequence_length, hidden_size)
        # attention_output[b][t][h]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第h个隐藏特征分量" 的注意力
        attention_probs.shape = (batch_size, num_attention_heads, sequence_length(T), sequence_length(S))
        # attention_probs[b][i][t][s]: 第b组序列, 第n个注意力, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性

        Args:
            `config (dict)`: 网络配置\n
        """
        super().__init__()
        self.hidden_size = config["hidden_size"]
        """隐藏特征维度"""
        self.num_attention_heads = config["num_attention_heads"]
        """注意力机制头数"""
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        """单头注意力机制的特征维度"""
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        """总特征维度"""
        self.qkv_bias = config["qkv_bias"]
        """偏置启用标志位"""
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        """注意力机制网络层"""
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        """dropout 层"""
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        """输出层"""
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
        """dropout 层"""

    def forward(self, x: torch.Tensor, output_attentions=False) -> tuple[torch.Tensor, torch.Tensor]:

        qkv = self.qkv_projection(x)
        # NOTE qkv.shape = (batch_size, sequence_length, all_head_size * 3)

        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # NOTE q | k | v.shape= (batch_size, sequence_length, all_head_size)

        batch_size, sequence_length, _ = query.size()

        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # NOTE q | k | v.shape= (batch_size, num_attention_heads, sequence_length, attention_head_size)

        # Calculate the attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # NOTE attention_scores.shape = (batch_size, num_attention_heads, sequence_length(T), sequence_length(S))
        # attention_scores[b][n][t][s]第b组序列, 第n个注意力, "Target序列索引t上Query" 关于 "Source序列索引s上Key"的相关性
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, value)
        # NOTE attention_output.shape = (batch_size, num_attention_heads, sequence_length(T), attention_head_size(V))
        # attention_output[b][n][t][v]: 第b组序列, 第n个注意力, "Target序列索引t上的Query" 对于 "Source序列第v个Value分量" 的注意力

        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)
        # transpose() --> shape = (batch_size,  sequence_length(Q), num_attention_heads, attention_head_size)
        # contiguous() --> deepcopy
        # view() --> shape = (batch_size, sequence_length(T), all_head_size)
        # NOTE attention_output[b][t][v]: 第b组序列, "Target序列索引t上的Query", 所有注意力头对"Source序列第v个Value分量" 的注意力

        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # attention_output.shape = (batch_size, sequence_length(T), hidden_size)
        # NOTE attention_output[b][t][h]: 第b组序列, "Target序列索引t上的Query" , 对于 "Source序列第h个隐藏特征分量" 的注意力

        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """多层感知机

    >>> output.shape = input.shape
    """

    def __init__(self, config: dict):
        """多层感知机

        >>> output.shape = input.shape

        Args:
            `config (dict)`: 网络配置\n
        """
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    包含一个多头自注意力机制的块

    >>> output.shape = input.shape
    attention_output, attention_probs = output
    attention_output.shape = (batch_size, sequence_length, hidden_size)
    # attention_output[b][t][h]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第h个隐藏特征分量" 的注意力
    attention_probs.shape = (batch_size, num_attention_heads, sequence_length(T), sequence_length(S))
    # attention_probs[b][i][t][s]: 第b组序列, 第i个单头, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性
    """

    def __init__(self, config: dict):
        """包含一个多头自注意力机制的块

        >>> output.shape = input.shape
        attention_output, attention_probs = output
        attention_output.shape = (batch_size, sequence_length, hidden_size)
        # attention_output[b][t][h]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第h个隐藏特征分量" 的注意力
        attention_probs.shape = (batch_size, num_attention_heads, sequence_length(T), sequence_length(S))
        # attention_probs[b][i][t][s]: 第b组序列, 第i个单头, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性

        Args:
            `config (dict)`: 网络配置\n
        """
        super().__init__()
        self.use_faster_attention: bool = config.get("use_faster_attention", False)
        """快速注意力机制启用标志位"""
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
            """自注意力层"""
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        """正则化层 1"""
        self.mlp = MLP(config)
        """多层感知机"""
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])
        """正则化层 2"""

    def forward(self, x: torch.Tensor, output_attentions=False) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """包含若干个 多头自注意力机制的 Encoder

    >>> output.shape = input.shape
    >>> attention_output, all_attentions = output
    attention_output.shape = (batch_size, sequence_length, hidden_size)
    # attention_output[b][t][h]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第h个隐藏特征分量" 的注意力
    for prob in all_attentions:
        prob.shape = (batch_size, num_attention_heads, sequence_length(T), sequence_length(S))
    # probs[b][i][t][s]: 第b组序列, 第i个单头, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性
    """

    def __init__(self, config: dict):
        """包含若干个 多头自注意力机制的 Encoder

        >>> output.shape = input.shape
        >>> attention_output, all_attentions = output
        attention_output.shape = (batch_size, sequence_length, hidden_size)
        # attention_output[b][t][h]: 第b组序列, "Target序列索引t上的Query" 对于 "Source序列第h个隐藏特征分量" 的注意力
        for prob in all_attentions:
            prob.shape = (batch_size, num_attention_heads, sequence_length(T), sequence_length(S))
        # probs[b][i][t][s]: 第b组序列, 第i个单头, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性

        Args:
            `config (dict)`: 网络配置\n
        """
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        """多头自注意力块"""
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x: torch.Tensor, output_attentions=False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    """用于图像分类的 vision Transformer

    >>> logits, all_attention = output
    logits.shape = (batch_size, num_classes)
    # logits[b][c]: 第b组序列class为c的概率
    for prob in all_attentions:
        prob.shape = (batch_size, num_attention_heads, sequence_length(T), sequence_length(S))
    # probs[b][i][t][s]: 第b组序列, 第i个单头, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性
    """

    def __init__(self, config: dict):
        """用于图像分类的 vision Transformer

        >>> logits, all_attention = output
        logits.shape = (batch_size, num_classes)
        # logits[b][0][c]: 第b组序列class为c的概率
        for prob in all_attentions:
            prob.shape = (batch_size, num_attention_heads, sequence_length(T), sequence_length(S))
        # probs[b][i][t][s]: 第b组序列, 第i个单头, "Target序列索引t上的Query" 对于 "Source序列索引s上的Key" 的相关性

        Args:
            `config (dict)`: 网络配置\n
        """
        super().__init__()
        self.config = config
        """网络配置"""
        self.image_size = config["image_size"]
        """输入图像大小"""
        self.hidden_size = config["hidden_size"]
        """隐藏特征维度"""
        self.num_classes = config["num_classes"]
        """分类标签数"""
        # Create the embedding module
        self.embedding = Embeddings(config)
        """图片编码层
        
        >>> output.shape = (batch_size, num_patches+1, hidden_size)
        """
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        """包含多头自注意力机制的 Encoder

        >>> logits, all_attentions = output
        logit.shape = (batch_size, 1, num_classes)

        """
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        """全连接层
        >>> output.shape[:-1] = input.shape[-1]
        output.shape[-1] = num_classes
        """
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # NOTE output.shape = (batch_size, num_patches+1, hidden_size)

        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # NOTE encoder_output.shape = (batch_size, num_patches+1, hidden_size)
        # prob in all_attentions: prob.shape = (batch_size, num_attention_heads, num_patches+1, attention_head_size)

        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        # NOTE logits.shape = (batch_size, num_classes)
        # logits[b][c]: 第b组序列，class为c的概率
        if not output_attentions:
            return logits
        else:
            return (logits, all_attentions)

    def _init_weights(self, module: nn.Module):
        """为 网络模型中的 `Conv2d`, `Linear`, `LayerNorm`, 和自定义的 `Embeddings` 以不同的方式初始化 权重和偏置

        Args:
            `module (nn.Module)`: 网络模型\n
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
