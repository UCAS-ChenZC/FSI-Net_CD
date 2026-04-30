import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.BaseNetworks import *
from models.help_funcs import TwoLayerConv2d, save_to_mat
import torch.nn.functional as F
import timm
import types
import math
from abc import ABCMeta, abstractmethod
from mmengine.model.weight_init import normal_init
from mmengine.logging import MMLogger
import logging
import warnings
from mmengine.runner import load_checkpoint as mmengine_load_checkpoint
from models.FFSwin import SwinTransformer
from models.channel_mapper import ChannelMapper
import torch.fft
from functools import partial
from models.MSDConv_SSFC import MSDConv_SSFC
import warnings
from einops import rearrange
from models.Decoder_head import SegFusion,UPerHead
warnings.filterwarnings('ignore')

# try:
#     from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d
# except ImportError as e:
ModulatedDeformConv2d = nn.Module
def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = MMLogger.get_instance(name='mmdet', log_file=log_file, log_level=log_level)

    return logger

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class First_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(First_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        x = self.conv(input)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.Conv = nn.Sequential(
            MSDConv_SSFC(in_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            MSDConv_SSFC(out_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.Conv(input)

class FSI_Former(nn.Module):

    def __init__(self, output_nc=2, decoder_softmax=False, embed_dim=256,ratio=0.5):
        super(FSI_Former, self).__init__()
        #Transformer Encoder
        self.embed_dims = [64,96, 192, 384, 768]
        # self.embed_dims = [32,64, 128, 320, 512]
        self.embedding_dim = embed_dim
        self.swin = SwinTransformer(
                        embed_dims=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.2,
                        attn_drop_rate=0.,
                        drop_path_rate=0.2,         #0.1 ?
                        patch_norm=True,
                        out_indices=(0, 1, 2, 3),
                        with_cp=False,
                        # convert_weights=True,
                        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'))
        
        self.DAFM1 =DAFM(96,input_size=(64,64))
        self.DAFM0 =DAFM(64,input_size=(128,128))
        self.DAFM2 =DAFM(192,input_size=(32,32))
        self.DAFM3 =DAFM(384,input_size=(16,16))
        self.DAFM4 =DAFM(768,input_size=(8,8))

        self.Conv1_First = First_DoubleConv(3, int(64 * ratio))
        # self.Conv1_2 = First_DoubleConv(3, int(64 * ratio))
        #2024.0111
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2_Double = DoubleConv(int(64 * ratio), int(128 * ratio))

        self.TDec_x2_V2   = SegFusion(input_transform='multiple_select', in_index=[0, 1, 2, 3,4], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = decoder_softmax, feature_strides=[2, 4, 8, 16,32])        #SegFusion Decoder_v4_SegFusion
        
        # self.Uper_Head = UPerHead(input_transform='multiple_select',in_index=[0, 1, 2, 3,4],align_corners=False,
        #             in_channels=self.embed_dims,pool_scales=(1, 2, 3, 6),channels=512,dropout_ratio=0.1,num_classes=2)

    def forward(self, x1, x2):
    #初步卷积后送入 Backbone进行特征提取
        x1 = self.Conv1_First(x1)
        # x1 = self.UFF1(x1)
        # x1 = self.FADConv1(x1)                    ### 尝试先对输入图像加入FADConv提取频域特征 ###
        x2 = self.Conv1_First(x2)
        # x2 = self.UFF1(x2)
        # x2 = self.FADConv1(x2)
        x1_temp_1 = x1
        x2_temp_1 = x2    

    #特征提取过程
        x1 = self.swin(x1)                          #x1 x2 为列表格式，含四个尺度的元素[96 192 384 768]
        x2 = self.swin(x2)

    #额外加入Decoder尺寸
        x1_temp_2 = self.Maxpool(x1_temp_1)
        x1_temp_2 = self.Conv2_Double(x1_temp_2)
        # x1_temp_2 = self.UFF2(x1_temp_2)
        # x1_temp_2 = self.FADConv2(x1_temp_2)

        x2_temp_2 = self.Maxpool(x2_temp_1)
        x2_temp_2 = self.Conv2_Double(x2_temp_2)
        # x2_temp_2 = self.UFF2(x2_temp_2)
        # x2_temp_2 = self.FADConv2(x2_temp_2)

        x1.insert(0,x1_temp_2)
        x2.insert(0,x2_temp_2)
        #x1.insert(0,x1_temp_1)
        #x2.insert(0,x2_temp_1)

    #在Decoder前进行三注意力融合
        x1[0] = self.DAFM0(x1[0])                           #DAFM注意力操作
        x1[1] = self.DAFM1(x1[1])
        x1[2] = self.DAFM2(x1[2])
        x1[3] = self.DAFM3(x1[3])
        x1[4] = self.DAFM4(x1[4])

        x2[0] = self.DAFM0(x2[0])
        x2[1] = self.DAFM1(x2[1])
        x2[2] = self.DAFM2(x2[2])
        x2[3] = self.DAFM3(x2[3])
        x2[4] = self.DAFM4(x2[4])

    #在Decoder前进行多个特征图的通道转换
        # [fx1, fx2] = [self.channelmapper(x1), self.channelmapper(x2)]
        [fx1, fx2] = [x1, x2]                         #self.Uper_Head

    #Decoder过程
        cp = self.TDec_x2_V2(fx1, fx2)          
        # cp = self.Uper_Head([fx1, fx2])
        # print(type(cp))
        # exit()
        return cp

    

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # 最大池化 & 平均池化
        max_out = self.max_pool(x).view(b, c)  # 变为 (b, c)
        avg_out = self.avg_pool(x).view(b, c)  # 变为 (b, c)
        # 经过 MLP 并求和
        max_out = self.mlp(max_out)
        avg_out = self.mlp(avg_out)
        out = max_out + avg_out
        # 通过 Sigmoid
        out = self.sigmoid(out).view(b, c, 1, 1)
        return out    

class SpatialAttention(nn.Module): #空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x

class DAFM(nn.Module):
    '''Dual Attention Fusion Module'''
    def __init__(self, channel, reduction=16, kernel_size=7,input_size=(128,128)):
        super(DAFM, self).__init__()
        self.channelattention = ChannelAttention(channel, reduction=reduction)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
        self.Conv3_1 = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=channel),
                    nn.ReLU())
        self.Conv3_2 = nn.Sequential(
                    nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=channel),
                    nn.ReLU())
        self.Conv3_3 = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=channel),
                    nn.ReLU())
        # self.dilated_conv_1 = nn.Sequential(
        #             nn.Conv2d(channel, channel,kernel_size=3, padding=2, dilation=2),
        #             nn.Conv2d(channel, channel,kernel_size=3, padding=4, dilation=4),
        #             nn.BatchNorm2d(num_features=channel),
        #             nn.ReLU())
        # self.dilated_conv_2 = nn.Sequential(
        #             nn.Conv2d(channel * 2, channel * 2,kernel_size=3, padding=2, dilation=2),
        #             nn.Conv2d(channel * 2, channel,kernel_size=3, padding=4, dilation=4),
        #             nn.BatchNorm2d(num_features=channel),
        #             nn.ReLU())
        # self.dilated_conv_3 = nn.Sequential(
        #             nn.Conv2d(channel, channel,kernel_size=3, padding=2, dilation=2),
        #             nn.Conv2d(channel, channel,kernel_size=3, padding=4, dilation=4),
        #             nn.BatchNorm2d(num_features=channel),
        #             nn.ReLU())
        self.Conv1 = nn.Conv2d(channel * 2,channel * 2,kernel_size=1)
        H,W = input_size
        # self.FAM = FAM_Module(channel,channel,shapes=H)
    def forward(self,x):
        
        x1 = x
        fc = x1 * self.channelattention(x1)     #Wc = self.channelattention(x1)
        fc = self.Conv3_1(fc)              #   dilated_conv    Conv3_1     FAM
        fs = fc * self.spatialattention(fc)     #Ws = self.spatialattention(fc)
        cat = torch.concat([x1,fs],dim=1)
        fd = cat * self.Conv1(cat)
        fd = self.Conv3_2(fd)
        # f_out_res = self.Conv3_3(f_out)
        f_out_res = self.Conv3_3(fd)
        f_out = fd + f_out_res
        x  = x + f_out
        return x



