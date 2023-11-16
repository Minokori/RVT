import torch
from torch import nn, optim
import torch.utils.data as torchdata
from utils._utils import save_experiment, save_checkpoint
from torcheval.metrics.functional import multiclass_f1_score
from RetNet import *
import logging
import datetime
imgcfg = ImgInputConfig(32)
netcfg = RetNetConfig(8 * 8, 4, 48, 128, 4, True)
clscfg = LabelOutputConfig(10)
logging.basicConfig(filename=f"./{datetime.date.today()}.log", level=logging.WARNING, encoding="utf-8")


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module, exp_name: str, device: torch.device, scheduler: optim.lr_scheduler.LRScheduler = None):
        """初始化训练器

        Args:
            `model (nn.Module)`: 训练模型\n
            `optimizer (optim.Optimizer)`: 优化器\n
            `loss_fn (nn.Module)`: 损失函数\n
            `exp_name (str)`: 实验名称\n
            `device (torch.device)`: cpu | cuda\n
        """
        self.model = model.to(device)
        """网络模型"""
        self.optimizer = optimizer
        """优化器"""
        self.scheduler = scheduler
        """动态损失函数调整"""
        self.loss_fn = loss_fn
        """损失函数"""
        self.exp_name = exp_name
        """实验名称"""
        self.device = device
        """cpu | cuda"""

    def train(self, trainloader: torchdata.DataLoader, testloader: torchdata.DataLoader, epochs: int, save_model_every_n_epochs=0):
        """训练模型

        Args:
            `trainloader (torchdata.DataLoader)`: 训练集DataLoader\n
            `testloader (torchdata.DataLoader)`: 测试集DataLoader\n
            `epochs (int)`: 训练步长\n
            `save_model_every_n_epochs (int, optional)`: 保存间隔. Defaults to 0.\n
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # TODO F1score
        F1score = []

        # Train the model
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss, test_f1 = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            # TODO
            F1score.append(test_f1)
            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Test F1: {test_f1}")

            if save_model_every_n_epochs > 0 and (i + 1) % save_model_every_n_epochs == 0 and (i + 1) != epochs:
                print(f"\tSave checkpoint at epoch : {i+1}")
                save_checkpoint(self.exp_name, self.model, i + 1)
        # Save the experiment
        # TODO
        save_experiment(self.exp_name, None, self.model, train_losses, test_losses, accuracies, F1score)

    def train_epoch(self, trainloader: torchdata.DataLoader) -> float:
        """单步训练, 返回损失

        Args:
            `trainloader (torchdata.DataLoader)`: 训练DataLoader\n

        Returns:
            `float`: 损失\n
        """
        self.model.train()
        total_loss = 0
        # DEBUG
        logging.debug(f"\n开始一次单步训练...\t{datetime.datetime.now().time()}")
        for batch in trainloader:
            # Move the batch to the device
            # DEBUG
            logging.debug(f"\t开始将图片加载到GPU上...\t{datetime.datetime.now().time()}")
            batch: list[torch.Tensor] = [t.to(self.device, non_blocking=True) for t in batch]
            images, labels = batch
            # images.shape = (b, c, h, w) | labels.shape = (b, cls)
            # DEBUG
            logging.debug(f"\t加载图片结束...\t{datetime.datetime.now().time()}")
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            # DEBUG
            logging.debug(f"\t开始计算损失...\t{datetime.datetime.now().time()}")
            # BUG 用 LMDB加载图片得到label没有独热编码
            loss: torch.Tensor = self.loss_fn(self.model(images), labels)
            # DEBUG
            logging.debug(f"\t计算损失结束...\t{datetime.datetime.now().time()}")
            # Backpropagate the loss

            # DEBUG
            logging.debug(f"\t开始反向传播...{datetime.datetime.now().time()}")
            loss.backward()
            logging.debug(f"\t反向传播结束...{datetime.datetime.now().time()}")
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        # DEBUG
        logging.debug(f"\n结束单步训练...\t{datetime.datetime.now().time()}")
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader: torchdata.DataLoader) -> tuple[float, float]:
        """评估模型

        Args:
            `testloader (torchdata.DataLoader)`: 测试集DataLoader\n

        Returns:
            `tuple[float, float]`: 准确率, 平均损失\n
        """
        self.model.eval()
        total_loss = 0
        # TODO
        total_f1 = 0

        correct = 0
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch: torch.Tensor = [t.to(self.device) for t in batch]
                images, labels = batch
                # NOTE images.shape = [batch_size, image.shape]
                # Get predictions
                logits = self.model(images)
                # NOTE logits.shape = (batch_size, num_classes)
                # Calculate the loss

                loss: torch.Tensor = self.loss_fn(logits, labels)

                # TODO
                f1score = multiclass_f1_score(logits, labels, num_classes=10)

                total_loss += loss.item() * len(images)
                # TODO
                total_f1 += f1score * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        # TODO
        avg_f1 = total_f1 / len(testloader.dataset)
        return accuracy, avg_loss, float(avg_f1)
