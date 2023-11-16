from torch import nn
from .gvt import ALTGVT, Block
from .vit import ViTForClassfication
import torchvision.models.mobilenetv3 as MobileNet
config = {
    "patch_size": 4,
    "hidden_size": 256,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 64,  # 4 * hidden_size
    "hidden_dropout_prob": 0.5,
    "attention_probs_dropout_prob": 0.5,
    "initializer_range": 0.02,
    "image_size": 64,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": False,
}

visiontransformer = ViTForClassfication(config)
altgvt = ALTGVT(config["image_size"],
                config["patch_size"],
                config["num_channels"],
                config["num_classes"],
                attn_drop_rate=config["attention_probs_dropout_prob"],
                drop_rate=config["hidden_dropout_prob"],
                drop_path_rate=0.5,
                block_cls=Block)
mobilenetv3 = MobileNet.mobilenet_v3_small()
mobilenetv3.classifier[3] = nn.Linear(1024, 10)
