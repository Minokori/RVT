import json
import os
from typing import Union

import torch


def save_experiment(experiment_name: str, config: dict, model: torch.nn.Module, train_losses: float, test_losses: float, accuracies: float, f1_score,base_dir="experiments") -> None:
    """保存实验配置文件, 性能指标, 网络模型到根目录

    + 根目录 `./experiments/{experiment_name}`
    + 配置文件 `~/config.json`
    + 性能指标 `~/metrics.json`
    + 网络模型 `~/model_final.pt`


    Args:
        `experiment_name (str)`: 实验名称\n
        `config (dict)`: 实验设置\n
        `model (torch.nn.Module)`: 网络模型\n
        `train_losses (float)`: 训练集损失\n
        `test_losses (float)`: 测试集损失\n
        `accuracies (float)`: 准确率\n
        `base_dir (str, optional)`: 根目录. Defaults to "experiments".\n
    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # Save the config
    if config is not None:
        config_file = os.path.join(outdir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

    # Save the metrics
    metrics_file = os.path.join(outdir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
            'Fl_scores' : f1_score,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name: str, model: torch.nn.Module, epoch: Union[int, str], base_dir="experiments") -> None:
    """保存网络检查点。

    保存到 `base_dir/experiment_name/model_{epoch}.pt`

    Args:
        `experiment_name (str)`: 实验名称\n
        `model (torch.nn.Module)`: 网络模型\n
        `epoch (int | str)`: 训练步次\n
        `base_dir (str, optional)`: 根目录. Defaults to "experiments".
    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)
