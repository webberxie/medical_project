# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

# Swin Transformer parameters



class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, zero_head=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head


        SWIN_PATCH_SIZE = 4
        SWIN_IN_CHANS = 3
        SWIN_EMBED_DIM = 96
        SWIN_DEPTHS = [1, 1, 1, 1]
        SWIN_DECODER_DEPTHS = [1, 1, 1, 1]
        SWIN_NUM_HEADS = [3, 6, 12, 24]
        SWIN_WINDOW_SIZE = 7
        SWIN_MLP_RATIO = 4.
        SWIN_QKV_BIAS = True
        SWIN_QK_SCALE = None
        SWIN_APE = False
        SWIN_PATCH_NORM = True
        SWIN_FINAL_UPSAMPLE = "expand_first"
        DROP_RATE = 0.5
        DROP_PATH_RATE = 0.5
        USE_CHECKPOINT = False

        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                patch_size=SWIN_PATCH_SIZE,
                                in_chans=SWIN_IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=SWIN_EMBED_DIM,
                                depths=SWIN_DEPTHS,
                                depths_decoder = SWIN_DECODER_DEPTHS,
                                num_heads=SWIN_NUM_HEADS,
                                window_size=SWIN_WINDOW_SIZE,
                                mlp_ratio=SWIN_MLP_RATIO,
                                qkv_bias=SWIN_QKV_BIAS,
                                qk_scale=SWIN_QK_SCALE,
                                drop_rate=DROP_RATE,
                                drop_path_rate=DROP_PATH_RATE,
                                ape=SWIN_APE,
                                patch_norm=SWIN_PATCH_NORM,
                                use_checkpoint=USE_CHECKPOINT)

    # 输入：B,C,H,W
    def forward(self, x):
        # 单通道复制成3通道
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
 