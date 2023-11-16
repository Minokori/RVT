# 神经网络函数

###
# region 库导入
import time
import os
import numpy
import torch
import torch.utils.data as data
import utils._data as _data
import shelve
from torch import nn
from IPython import display
from matplotlib import pyplot
import torchvision
import torchvision.datasets as visiondata
# endregion


###
def train(net: nn.Module,
          train_iter: data.DataLoader, val_iter: data.DataLoader,
          num_epochs: int, lr: float,
          device: torch.device,
          net_name: str) -> None:
    """神经网络训练函数

    Args:
        net (nn.Module): 神经网络
        train_iter (data.DataLoader): 训练集数据加载器
        test_iter (data.DataLoader): 测试集数据加载器
        num_epochs (int): 训练步长
        lr (float): 学习率
        device (torch.device): 训练设备
        net_name (str): 网络名称，用于保存网络
    """

    def init_weights(m: nn.Module) -> None:
        """网络权重初始化。
        将网络中的 全连接层 与 卷积层 中的权重以 xavier_uniform_ 方式初始化

        Args:
            m (nn.Sequential): 神经网络
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # region 初始化
    net_saved_path = os.path.join(_data.NET_PATH, net_name+".pt")  # 网络保存的路径

    info = []  # 初始化消息列表
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'val acc'],
                        info=info)  # 初始化绘图
    timer = Timer()
    # endregion

    # region 初始化网络
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 优化器
    loss = nn.CrossEntropyLoss()  # 损失函数
    num_batches = len(train_iter)  # 训练集小批量的大小
    net.apply(init_weights)  # 初始化随机权重

    # endregion

    # region 初始化或加载网络
    if os.path.exists(net_saved_path):  # 加载网络
        # region 加载网络
        checkpoint = torch.load(net_saved_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        # endregion
        # region 加载绘图与计时器
        # saved_info = shelve.open(info_saved_path)
        # animator.X = saved_info["saved_X"]
        # animator.Y = saved_info["saved_Y"]
        # animator.info = saved_info["saved_info"]
        # timer.times = saved_info["saved_times"]
        # timer = saved_info["timer"]
        # saved_info.close()
        # endregion
        print('加载 epoch {} 成功 '.format(start_epoch))
    else:  # 初始化网络参数
        start_epoch = 0
        info = ["训练设备: "+str(torch.cuda.get_device_name()), ]  # 训练状态信息列表
        info.append("网络权重初始化")
    # endregion

    print_msg(info)  # 输出训练状态信息
    net.to(device)  # 将网络移动到GPU上

    # region 迭代训练
    for epoch in range(start_epoch, num_epochs):

        metric = Accumulator(3)  # 设置累加器
        net.cuda().train()  # 将网络设置为训练模式

        # region 训练过程
        for i, (img, label) in enumerate(train_iter):  # 枚举训练集数据加载器中的图像与标签

            timer.start()  # 计时器开始（只计训练模型消耗的时间）

            optimizer.zero_grad()  # 梯度置零(将 loss 关于 weight 的导数变成 0)
            img, label = img.to(device), label.to(device)  # 将数据移动至 GPU 上
            label_hat = net(img)  # 向网络中输入图片得到预测标签
            l = loss(label_hat, label)  # 计算预测标签与真实标签之间的 loss
            l.backward()  # 反向传播
            optimizer.step()  # 执行一次优化过程

            # 计算预测准确率
            with torch.no_grad():
                metric.add(
                    l * img.shape[0],  # 累加器中第一个值为 计算损失 loss
                    accuracy(label_hat, label),  # 累加器中第二个值为 计算正确的数量
                    img.shape[0])  # 累加器中第三个值为 样本数量

            timer.stop()  # 计时器停止

            train_loss = metric[0] / metric[2]  # 计算训练损失
            train_acc = metric[1] / metric[2]  # 计算网络在 训练集 上的准确率

            # 绘制 训练集 损失和准确率
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                # 绘制 训练集 损失和准确率
                animator.add(epoch + (i + 1) / num_batches,
                             (train_loss, train_acc, None))
        # endregion
        val_acc = evaluate_accuracy_gpu(net, val_iter)  # 计算网络在 验证集 上的准确率
        animator.add(epoch + 1, (None, None, val_acc))  # 绘制 验证集 准确率
        state = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch}  # 模型参数
        # region 设置检查点
        if (epoch+1) % 5 == 0 or epoch == num_batches-1:  # 每五步及最后一步执行保存操作
            # 保存模型
            torch.save(state, net_saved_path)
            # 更新消息列表
            info.append("训练执行到 "+str(epoch+1)+" 次迭代，已保存检查点")
            # 保存绘图与计时器
            # info_save = shelve.open(info_saved_path)
            # info_save["saved_X"] = animator.X
            # info_save["saved_Y"] = animator.Y
            # info_save["saved_info"] = animator.info
            # info_save["saved_times"] = timer.times
            # info_save.close()
        # endregion
        # region 输出结果
    print(f'训练损失/train loss {train_loss:.3f}, 训练集准确率/train acc {train_acc:.3f}, '
          f'验证集准确率/val acc {val_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    # endregion
# endregion
# region 训练函数子函数


def evaluate_accuracy_gpu(net: nn.Module,
                          data_iter: data.DataLoader,
                          device=None) -> float:
    """使用GPU计算网络在数据集上的准确率

    Args:
        net (nn.Module): 神经网络
        data_iter (data.DataLoader): 数据集对应的数据加载器
        device (torch.device, optional): 计算所使用的设备. Defaults to None.

    Returns:
        float: 预测的准确率
    """
    if isinstance(net, nn.Module):  # 若网络有效
        net.eval()  # 将网络设置为评估模式

        # 使用GPU
        if not device:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)  # 设置累加计数器

    with torch.no_grad():
        for img, label in data_iter:  # 枚举数据加载器中的图片与标签

            # 将图片移动到 GPU 上
            if isinstance(img, list):
                img = [x.to(device) for x in img]
            else:
                img = img.to(device)

            # 将标签移动到GPU上
            label = label.to(device)

            # 将预测正确的数量，样本总数量储存在累加器中
            metric.add(accuracy(net(img), label), label.numel())

    return metric[0] / metric[1]  # 输出正确率


def accuracy(label_hat: torch.Tensor, label: torch.Tensor) -> float:
    """计算预测正确的数量

    Args:
        label_hat (torch.Tensor): 预测标签
        label (torch.Tensor): 实际标签

    Returns:
        float: 预测正确的数量
    """
    if len(label_hat.shape) > 1 and label_hat.shape[1] > 1:  # 训练第一步不进行计算
        # 计算预测标签
        label_hat = label_hat.argmax(axis=1)

    pred = label_hat.type(label.dtype) == label
    # 先将 预测标签 的数据类型转换为与 实际标签 相同
    # 再判断其值是否与 实际标签相同
    # 相同返回1，不同返回0

    return float(pred.type(label.dtype).sum())  # 输出相同的数量


def try_gpu(i=0) -> torch.device:
    """尝试使用GPU进行训练

    Args:
        i (int, optional): 训练设备编号. Defaults to 0.

    Returns:
        torch.device: 返回训练使用的设备
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def print_msg(info: list) -> None:
    """打印消息

    Args:
        info (list): 消息列表
    """
    for msg in info:
        print(msg)

# endregion

# region 训练函数子类


class Accumulator:
    """用于计数的累加器"""

    def __init__(self, n: int) -> None:
        """用于计数的累加器

        Args:
            n (int): 计数器个数
        Method:
            add: 向累加器添加数据
            reset: 将存在的值置零
        """
        self.data = [0.0] * n

    def add(self, *args) -> None:
        """向累加器添加数据，数据将依次存放在`Accumulator.data[index]`中
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self) -> None:
        """将存在的值置零
        """
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int) -> int:
        """直接调用`Accumulator`的返回值

        Args:
            idx (int): 索引值

        Returns:
            int: 累加器对应标签储存的数值
        """
        return self.data[idx]


class Timer:
    """用于记录运算时间的计时器"""

    def __init__(self) -> None:
        """用于记录运算时间的计时器

        Method:
        start:开始计时
        stop:停止计时，并返回从上一次`Timer.start()`开始的时长
        avg:返回计时器每次计时平均时长
        sum:返回计时器计时总时长
        cumsum:返回一个列表，列表值是计时次数对应的累计总时长
        """
        self.times = []
        self.start()

    def start(self) -> None:
        """开始计时"""
        self.tik = time.time()

    def stop(self) -> float:
        """停止计时，并返回从上一次`Timer.start()`开始的时长

        Returns:
            float: 从上一次`Timer.start()`开始的时长
        """
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self) -> float:
        """返回计时器每次计时平均时长

        Returns:
            float: 每次计时平均时长
        """
        return sum(self.times) / len(self.times)

    def sum(self) -> float:
        """返回计时器计时总时长

        Returns:
            float: 计时器计时总时长
        """
        return sum(self.times)

    def cumsum(self) -> list:
        """返回一个列表，列表值是计时次数对应的累计总时长
            (从第一次`Timer.start()`开始的总时长)

        Returns:
            list: 计时次数对应的累计总时长
        """
        return numpy.array(self.times).cumsum().tolist()


class Animator:
    """动态绘制折线图"""

    def __init__(self,
                 xlabel=None, ylabel=None,
                 legend=None,
                 xlim=None, ylim=None,
                 xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'),
                 nrows=1, ncols=1,
                 figsize=(7, 5),
                 info=[]) -> None:
        """动态绘制折线图

        Args:
            xlabel (str, optional): X轴名称. Defaults to None.
            ylabel (str, optional): Y轴名称. Defaults to None.
            legend (list, optional): 图例. Defaults to None.
            xlim (tuple, optional): X轴坐标范围. Defaults to None.
            ylim (tuple, optional): Y轴坐标范围. Defaults to None.
            xscale (str, optional): X轴缩放方式. Defaults to 'linear'.
            yscale (str, optional): Y轴缩放方式. Defaults to 'linear'.
            fmts (tuple, optional): 线条格式. Defaults to ('-', 'm--', 'g-.', 'r:').
            nrows (int, optional): 子绘图网格的行数. Defaults to 1.
            ncols (int, optional): 子绘图网格的列数. Defaults to 1.
            figsize (tuple, optional): 子绘图网格的图像大小. Defaults to (14, 10).
            info (list): 输出信息
        Method:
            add:向图中添加多个数据点
        """

        if legend is None:
            legend = []
        self.info = info
        display.set_matplotlib_formats('svg')  # 将图片设置为 svg 格式

        # 设置绘制多子图中 子绘图网格的行数，列数，图像大小
        self.fig, self.axes = pyplot.subplots(
            nrows, ncols, figsize=figsize)

        # 若仅有一个子图，则调整绘图大小
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 保存入参设置
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y) -> None:
        """向图中添加多个数据点

        Args:
            x: 数据点对应的横坐标
            y: 数据点对应的纵坐标
        """

        # 将 x，y 中的数据点添加到 Animator.X 与 Animator.Y 中
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        self.axes[0].cla()  # 清除当前轴

        # 绘制（x，y）
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)

        # 设置图像格式
        self.config_axes()

        # 显示图像
        display.display(self.fig)
        print_msg(self.info)
        display.clear_output(wait=True)


def set_axes(axes,
             xlabel: str, ylabel: str,
             xlim: tuple, ylim: tuple,
             xscale: str, yscale: str,
             legend: list) -> None:
    """设置图像格式

    Args:
        axes (_type_): 坐标系
        xlabel (str): X轴名称
        ylabel (str): Y轴名称
        xlim (tuple): X轴坐标范围
        ylim (tuple): Y轴坐标范围
        xscale (str): X轴缩放类型
        yscale (str): Y轴缩放类型
        legend (list): 图例
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
# endregion


# region 图像预处理
def TRANS(size: int, is_VAL=False) -> torchvision.transforms:
    normalize_imgnet = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
    if is_VAL is False:
        return torchvision.transforms.Compose(
            [torchvision.transforms.Resize((size, size)),
             torchvision.transforms.RandomHorizontalFlip(p=0.5),
             torchvision.transforms.RandomVerticalFlip(p=0.5),
             torchvision.transforms.RandomRotation(15),
             torchvision.transforms.ToTensor(),
             ])
    else:

        return torchvision.transforms.Compose(
            [torchvision.transforms.Resize((int(size*1.2), int(size*1.2))),
             torchvision.transforms.CenterCrop((size, size)),
             torchvision.transforms.ToTensor()])
# endregion

# region 从图片数据集中获取小批量数据


def GetBatch(dataset: visiondata.ImageFolder,
             batch_size: int) -> data.DataLoader:
    """从数据集中取得小批量

    Args:
        dataset (visiondata.ImageFolder): 数据集
        batch_size (int): 批量大小

    Returns:
        data.DataLoader: 一个包含`batch_size`个样本的`DataLoader`实例
    """
    batch = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    return batch
# endregion

# region 检查小批量输入


def print_data_info(batch: data.DataLoader,
                    ) -> None:
    """打印第一个小批量

    Args:
        train_batch (data.DataLoader): 训练用小批量
        test_batch (data.DataLoader): 测试用小批量
    """

    def print_data(data_batch: data.DataLoader) -> None:
        for batch_i, img in enumerate(data_batch):
            if batch_i == 0:
                # img代表一个列表：img[0]代表数据，img[1]代表标签
                print(img[0].type())
                fig = pyplot.figure(figsize=(200, 100))
                grid = torchvision.utils.make_grid(
                    img[0], nrow=16, normalize=False)
                # [C x H x W] 改变成 [H × W × C]
                pyplot.imshow(grid.numpy().transpose((1, 2, 0)), aspect="auto")
                pyplot.show()
            break
    print("items_type:")
    print_data(batch)
# endregion


# region 检查网络结构

def print_net_info(net: nn.Module, size: int, channel=3) -> None:
    """打印输出网络结构

    Args:
        net (nn.Module): 神经网络实例
        size (int): 输入图片大小
        channel (int, optional): 输入图片通道数. Defaults to 3.
    """
    T = torch.randn(size=(1, channel, size, size))
    for blk in net:
        T = blk(T)
        print(blk.__class__.__name__, "\toutput shape:\t", T.shape)
# endregion
