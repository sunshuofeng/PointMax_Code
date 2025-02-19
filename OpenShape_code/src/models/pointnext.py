import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "PointNeXt"))

from .PointNeXt.openpoints.models import build_model_from_cfg
from .PointNeXt.openpoints.utils import EasyConfig


class PointNeXt(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        scaled = [None, 'pointnext-s.yaml', 'pointnext-l.yaml', 'pointnext-xl.yaml','pointnext-pointmax.yaml'][cfg.model.scaling]
        subcfg = EasyConfig()
        subcfg.load(os.path.join(os.path.dirname(__file__), "pointnext_configs", scaled))
        subcfg = subcfg.model
        subcfg.encoder_args.in_channels = cfg.model.in_channel
        subcfg.cls_args.num_classes = cfg.model.out_channel
        self.net = build_model_from_cfg(subcfg)

    def forward(self, xyz, features: torch.Tensor,label=None):
        global_feat,all_loss = self.net.encoder.forward_cls_feat(xyz, features.transpose(-1, -2).contiguous(),label)
        if self.training:
            return self.net.prediction(global_feat),all_loss
        else:
            return self.net.prediction(global_feat)


def make(cfg):
    return PointNeXt(cfg)
