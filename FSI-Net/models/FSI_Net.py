import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
# from models.ChangeFormerBaseNetworks import *
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
from models.channel_mapper import ChannelMapper
import torch.fft
from functools import partial
from models.MSDConv_SSFC import MSDConv_SSFC
import warnings
from einops import rearrange
from models.Decoder_head import SegFusion,UPerHead
from models.RMT import FSIFormer
from models.FSI_Former import RMT_Freq
# from models.RME_Fre_127 import RMT_Freq
from einops import repeat
from timm.models.layers import DropPath
from typing import Callable
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
        # self.swin = SwinTransformer(
        #                 embed_dims=96,
        #                 depths=[2, 2, 6, 2],
        #                 num_heads=[3, 6, 12, 24],
        #                 window_size=7,
        #                 mlp_ratio=4,
        #                 qkv_bias=True,
        #                 qk_scale=None,
        #                 drop_rate=0.2,
        #                 attn_drop_rate=0.,
        #                 drop_path_rate=0.2,         #0.1 ?
        #                 patch_norm=True,
        #                 out_indices=(0, 1, 2, 3),
        #                 with_cp=False,
        #                 # convert_weights=True,
        #                 init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'))
        self.encoder = FSIFormer(
                        embed_dims=[96, 192, 384, 768], 
                        depths=[2, 2, 6, 2], 
                        num_heads=[3, 6, 12, 24],
                        init_values=[1, 1, 1, 1],  
                        heads_ranges=[3, 3, 3, 3], 
                        mlp_ratios=[3, 3, 3, 3], 
                        drop_path_rate=0.1, 
                        norm_layer=nn.LayerNorm, 
                        patch_norm=True, 
                        use_checkpoint=False, 
                        chunkwise_recurrents=[True, True, False, False], 
                        projection=1024,
                        layerscales=[False, False, False, False], 
                        layer_init_values=1e-6, norm_eval=True)
                        # convert_weights=True,
                        # init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'))
        # self.swin = RMT_Freq(
        #                 embed_dims=[96, 192, 384, 768], 
        #                 depths=[2, 2, 6, 2], 
        #                 num_heads=[3, 6, 12, 24],
        #                 init_values=[1, 1, 1, 1], 
        #                 heads_ranges=[3, 3, 3, 3], 
        #                 mlp_ratios=[4, 4, 4, 4], 
        #                 drop_path_rate=0.1, 
        #                 norm_layer=nn.LayerNorm, 
        #                 patch_norm=True, 
        #                 use_checkpoint=False, 
        #                 chunkwise_recurrents=[True, True, False, False], 
        #                 projection=1024,
        #                 layerscales=[False, False, False, False], 
        #                 # NEW
        #                 use_frequencies=[True, True, False, False],
        #                 freq_base_resolutions=[(64,64),(32,32),(16,16),(8,8)],
        #                 freq_energy_thresh=0.25,
        #                 freq_residual_scale=0.5,
        #                 head_scale_range=(0.5, 1.5),
        #                 norm_eval=True)
        #                 # layer_init_values=1e-6, norm_eval=True)
        # self.swin = RMT_Freq(
        #                 embed_dims=[96, 192, 384, 768], 
        #                 depths=[2, 2, 6, 2], 
        #                 num_heads=[3, 6, 12, 24],
        #                 init_values=[1, 1, 1, 1], 
        #                 heads_ranges=[3, 3, 3, 3], 
        #                 mlp_ratios=[4, 4, 4, 4], 
        #                 drop_path_rate=0.1, 
        #                 norm_layer=nn.LayerNorm, 
        #                 patch_norm=True, 
        #                 use_checkpoint=False, 
        #                 chunkwise_recurrents=[True, True, False, False], 
        #                 projection=1024,
        #                 layerscales=[False, False, False, False], 
        #                 # NEW
        #                 use_frequencies=[True, True, False, False],
        #                 num_filters=4,
        #                 lowfreq_sigma=0.35)
        
        # self.DAFM1 =DAFM(96,input_size=(64,64))
        # self.DAFM0 =DAFM(64,input_size=(128,128))
        # self.DAFM2 =DAFM(192,input_size=(32,32))
        # self.DAFM3 =DAFM(384,input_size=(16,16))
        # self.DAFM4 =DAFM(768,input_size=(8,8))  
          
        # self.DAFM1 =DAFM(96)
        # self.DAFM0 =DAFM(64)
        # self.DAFM2 =DAFM(192)
        # self.DAFM3 =DAFM(384)
        # self.DAFM4 =DAFM(768)
        
        self.DAFM1 =DAFM_Dy(96)
        self.DAFM0 =DAFM_Dy(64)
        self.DAFM2 =DAFM_Dy(192)
        self.DAFM3 =DAFM_Dy(384)
        self.DAFM4 =DAFM_Dy(768)

        self.Conv1_First = First_DoubleConv(3, int(64 * ratio))
        # self.Conv1_2 = First_DoubleConv(3, int(64 * ratio))
        #2024.0111
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2_Double = DoubleConv(int(64 * ratio), int(128 * ratio))

        self.decoder   = SegFusion(input_transform='multiple_select', in_index=[0, 1, 2, 3,4], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = decoder_softmax, feature_strides=[2, 4, 8, 16,32])        #SegFusion Decoder_v4_SegFusion
        
        # self.Uper_Head = UPerHead(input_transform='multiple_select',in_index=[0, 1, 2, 3,4],align_corners=False,
        #             in_channels=self.embed_dims,pool_scales=(1, 2, 3, 6),channels=512,dropout_ratio=0.1,num_classes=2)
        self.tip_vis = None

    def forward(self, x1, x2):
    #初步卷积后送入 Backbone进行特征提取
        x1 = self.Conv1_First(x1)
        x2 = self.Conv1_First(x2)
        x1_temp_1 = x1
        x2_temp_1 = x2    
    #特征提取过程
        x1 = self.encoder(x1)                          #x1 x2 为列表格式，含四个尺度的元素[96 192 384 768]
        x2 = self.encoder(x2)

    #额外加入Decoder尺寸
        x1_temp_2 = self.Maxpool(x1_temp_1)
        x1_temp_2 = self.Conv2_Double(x1_temp_2)

        x2_temp_2 = self.Maxpool(x2_temp_1)
        x2_temp_2 = self.Conv2_Double(x2_temp_2)

        x1 = [x1_temp_2, *x1]
        x2 = [x2_temp_2, *x2]
        # self.tip_vis = x2

    #在Decoder前进行特征细化
        x1[0] = self.DAFM0(x1[0])                      
        x1[1] = self.DAFM1(x1[1])
        x1[2] = self.DAFM2(x1[2])
        x1[3] = self.DAFM3(x1[3])
        x1[4] = self.DAFM4(x1[4])

        x2[0] = self.DAFM0(x2[0])
        x2[1] = self.DAFM1(x2[1])
        x2[2] = self.DAFM2(x2[2])
        x2[3] = self.DAFM3(x2[3])
        x2[4] = self.DAFM4(x2[4])

        # x1[0] = self.TIP0(x1[0])                           #TIP注意力操作
        # x1[1] = self.TIP1(x1[1])
        # x1[2] = self.TIP2(x1[2])
        # x1[3] = self.TIP3(x1[3])
        # x1[4] = self.TIP4(x1[4])
        
        # x2[0] = self.TIP0(x2[0])
        # x2[1] = self.TIP1(x2[1])
        # x2[2] = self.TIP2(x2[2])
        # x2[3] = self.TIP3(x2[3])
        # x2[4] = self.TIP4(x2[4])
        
    #Decoder过程
        [fx1, fx2] = [x1, x2]
        cp = self.decoder(fx1, fx2)          
        # cp = self.Uper_Head([fx1, fx2])
        # print(type(cp))
        # exit()
        return cp

    

# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(ChannelAttention, self).__init__()
#         self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction, in_channels, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, h, w = x.shape
#         # 最大池化 & 平均池化
#         max_out = self.max_pool(x).view(b, c)  # 变为 (b, c)
#         avg_out = self.avg_pool(x).view(b, c)  # 变为 (b, c)
#         # 经过 MLP 并求和
#         max_out = self.mlp(max_out)
#         avg_out = self.mlp(avg_out)
#         out = max_out + avg_out
#         # 通过 Sigmoid
#         out = self.sigmoid(out).view(b, c, 1, 1)
#         return out    

# class SpatialAttention(nn.Module): #空间注意力机制
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         x = self.sigmoid(x)
#         return x

# class DAFM(nn.Module):
#     '''Dual Attention Fusion Module'''
#     def __init__(self, channel, reduction=16, kernel_size=7,input_size=(128,128)):
#         super(DAFM, self).__init__()
#         self.channelattention = ChannelAttention(channel, reduction=reduction)
#         self.spatialattention = SpatialAttention(kernel_size=kernel_size)
#         self.Conv3_1 = nn.Sequential(
#                     nn.Conv2d(channel, channel, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(num_features=channel),
#                     nn.ReLU())
#         self.Conv3_2 = nn.Sequential(
#                     nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(num_features=channel),
#                     nn.ReLU())
#         self.Conv3_3 = nn.Sequential(
#                     nn.Conv2d(channel, channel, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(num_features=channel),
#                     nn.ReLU())
#         # self.dilated_conv_1 = nn.Sequential(
#         #             nn.Conv2d(channel, channel,kernel_size=3, padding=2, dilation=2),
#         #             nn.Conv2d(channel, channel,kernel_size=3, padding=4, dilation=4),
#         #             nn.BatchNorm2d(num_features=channel),
#         #             nn.ReLU())
#         # self.dilated_conv_2 = nn.Sequential(
#         #             nn.Conv2d(channel * 2, channel * 2,kernel_size=3, padding=2, dilation=2),
#         #             nn.Conv2d(channel * 2, channel,kernel_size=3, padding=4, dilation=4),
#         #             nn.BatchNorm2d(num_features=channel),
#         #             nn.ReLU())
#         # self.dilated_conv_3 = nn.Sequential(
#         #             nn.Conv2d(channel, channel,kernel_size=3, padding=2, dilation=2),
#         #             nn.Conv2d(channel, channel,kernel_size=3, padding=4, dilation=4),
#         #             nn.BatchNorm2d(num_features=channel),
#         #             nn.ReLU())
#         self.Conv1 = nn.Conv2d(channel * 2,channel * 2,kernel_size=1)
#         H,W = input_size
#         # self.FAM = FAM_Module(channel,channel,shapes=H)
#     def forward(self,x):
        
#         x1 = x
#         fc = x1 * self.channelattention(x1)     #Wc = self.channelattention(x1)
#         fc = self.Conv3_1(fc)              #   dilated_conv    Conv3_1     FAM
#         fs = fc * self.spatialattention(fc)     #Ws = self.spatialattention(fc)
#         cat = torch.concat([x1,fs],dim=1)
#         fd = cat * self.Conv1(cat)
#         fd = self.Conv3_2(fd)
#         # f_out_res = self.Conv3_3(f_out)
#         f_out_res = self.Conv3_3(fd)
#         f_out = fd + f_out_res
#         x  = x + f_out
#         return x

class DynamicSpatialAttentionMap(nn.Module):
    """
    Dynamic Spatial Attention (map version):
    - per-sample dynamic kernel generated from input feature
    - use avg+max spatial descriptors (2-channel), like CBAM
    - output attention map: (B, 1, H, W)
    """
    def __init__(self, in_channels, kernel_size=7, reduction=16,
                 use_max=True, rescale_to_one=False, zero_init=False):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.kernel_size = kernel_size
        self.use_max = use_max
        self.stat_ch = 2 if use_max else 1

        hidden = max(8, in_channels // reduction)

        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            # generate (stat_ch * k*k) parameters per sample
            nn.Conv2d(hidden, self.stat_ch * kernel_size * kernel_size, kernel_size=1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.rescale_to_one = rescale_to_one

        # 可选：更稳定的初始化（注意：如果用 rescale_to_one=True 更合适）
        if zero_init:
            nn.init.zeros_(self.kernel_generator[-1].weight)
            nn.init.zeros_(self.kernel_generator[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel_size

        # kernels: (B, stat_ch*k*k, 1, 1) -> (B, stat_ch, k, k)
        kernels = self.kernel_generator(x).view(B, self.stat_ch, k, k)

        # spatial descriptors
        avg_out = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        if self.use_max:
            max_out, _ = x.max(dim=1, keepdim=True)
            stats = torch.cat([avg_out, max_out], dim=1)  # (B,2,H,W)
        else:
            stats = avg_out  # (B,1,H,W)

        # group conv trick: treat each sample as a group
        stats = stats.view(1, B * self.stat_ch, H, W)
        att = F.conv2d(stats, weight=kernels, padding=k // 2, groups=B)  # (1,B,H,W)
        att = att.view(B, 1, H, W)
        att = self.sigmoid(att)

        # 可选：把门控中心移到 1（避免初期把特征整体压到 0.5）
        # att in [0,2]，当 conv 输出为 0 时，sigmoid=0.5 => att=1
        if self.rescale_to_one:
            att = att * 2.0

        return att
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.avg_pool(x).view(b, c)
        out = self.mlp(max_out) + self.mlp(avg_out)
        out = self.sigmoid(out).view(b, c, 1, 1)
        return out
    
class SoftmaxFusion(nn.Module):
    """Softmax-gated fusion for two branches x and y."""
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=True)

    def forward(self, x, y):
        B, C, H, W = x.shape
        w = self.gate(torch.cat([x, y], dim=1))     # (B,2C,H,W)
        w = w.view(B, 2, C, H, W)
        w = torch.softmax(w, dim=1)
        return w[:, 0] * x + w[:, 1] * y


# class DAFM(nn.Module):
#     """
#     Recommended:
#     - Keep your CA
#     - Replace SA with DynamicSpatialAttentionMap
#     - Replace cat*Conv(cat) with softmax fusion (more paper-friendly)
#     - Add LayerScale gamma (stable behind Swin/RMT)
#     """
#     def __init__(self, channel, reduction=16, kernel_size=7, layerscale_init=1e-2):
#         super().__init__()
#         self.ca = ChannelAttention(channel, reduction=reduction)
#         self.sa = DynamicSpatialAttentionMap(
#             in_channels=channel,
#             kernel_size=kernel_size,
#             reduction=reduction,
#             use_max=True,
#             # 建议打开：让注意力初值更接近 1（不压缩特征）
#             rescale_to_one=True,
#             # 配合 rescale_to_one=True 更稳
#             zero_init=True
#         )

#         self.pre = nn.Sequential(
#             nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(inplace=True),
#         )

#         self.fuse = SoftmaxFusion(channel)

#         self.refine = nn.Sequential(
#             nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channel),
#         )

#         # LayerScale：插 transformer 后一般更安全
#         self.gamma = nn.Parameter(torch.ones(1, channel, 1, 1) * layerscale_init)

#     def forward(self, x):
#         x1 = x

#         fc = x1 * self.ca(x1)
#         fc = self.pre(fc)

#         fs = fc * self.sa(fc)

#         fused = self.fuse(x1, fs)

#         out = self.refine(fused)
#         return x1 + self.gamma * out


class DAFM_Dy(nn.Module):
    """
    Minimal change from your DAFM:
    - SpatialAttention -> DynamicSpatialAttentionMap
    - other parts unchanged
    """
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.channelattention = ChannelAttention(channel, reduction=reduction)
        self.spatialattention = DynamicSpatialAttentionMap(
            in_channels=channel,
            kernel_size=kernel_size,
            reduction=reduction,
            use_max=True,
            rescale_to_one=False,  # 保持和你原本 0~1 门控一致
            zero_init=False
        )

        self.Conv3_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.Conv3_2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.Conv3_3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.Conv1 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = x
        fc = x1 * self.channelattention(x1)
        fc = self.Conv3_1(fc)

        # dynamic spatial attention map
        fs = fc * self.spatialattention(fc)

        cat = torch.cat([x1, fs], dim=1)
        fd = cat * self.Conv1(cat)
        fd = self.Conv3_2(fd)

        f_out_res = self.Conv3_3(fd)
        f_out = fd + f_out_res

        x = x + f_out
        return x

class CEB(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0)
        )
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):# 输入 BHWC 输出 BHWC
        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        x = self.conv(x)
        skip = x
        x1, x2 = torch.split(x, C // 2, dim=1)
        avg_out = self.mlp1(self.avg_pool(x1))
        max_out = self.mlp2(self.max_pool(x2))
        y1 = self.sigmoid1(avg_out)
        y2 = self.sigmoid2(max_out)
        z = torch.cat((x1 * y1, x2 * y2), dim=1)
        perm = torch.randperm(C)
        z = z[:, perm, :, :]
        z = z + skip
        z = z.permute(0, 2, 3, 1).contiguous()
        return z


class FEB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.select1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.select2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):# 输入 BHWC 输出 BHWC
        B, H, W, C = x.shape
        skip = x
        x = x.permute(0, 3, 1, 2).contiguous()
        y = torch.fft.rfft2(x) + 1e-8
        a = torch.abs(y)
        p = torch.angle(y)
        a = self.select1(a)
        p = self.select2(p)
        real = a * torch.cos(p)
        imag = a * torch.sin(p)
        out = torch.complex(real, imag) + 1e-8
        out = torch.fft.irfft2(out, s=(H, W), norm='backward') + 1e-8
        out = torch.abs(out) + 1e-8

        out = out.permute(0, 2, 3, 1).contiguous()
        out = out + skip

        return out

# class SSM2D_MB(nn.Module):
#     def __init__(
#             self,
#             d_model,
#             d_state=16,
#             expand=2.,
#             dt_rank="auto",
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             dropout=0.,
#             device=None,
#             dtype=None,
#             **kwargs,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#         self.x_proj = (
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#         )
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
#         del self.x_proj
#         self.dt_projs = (
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#         )
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
#         del self.dt_projs
#         self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
#         self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
#         self.selective_scan = selective_scan_fn
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else None

#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError

#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)

#         dt_proj.bias._no_reinit = True

#         return dt_proj

#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)
#         if copies > 1:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log

#     @staticmethod
#     def D_init(d_inner, copies=1, device=None, merge=True):
#         D = torch.ones(d_inner, device=device)
#         if copies > 1:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)
#         D._no_weight_decay = True
#         return D

#     def forward_core(self, x: torch.Tensor):
#         B, C, H, W = x.shape
#         L = H * W
#         K = 4
#         x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
#                              dim=1).view(B, 2, -1, L)
#         xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#         xs = xs.float().view(B, -1, L)
#         dts = dts.contiguous().float().view(B, -1, L)
#         Bs = Bs.float().view(B, K, -1, L)
#         Cs = Cs.float().view(B, K, -1, L)
#         Ds = self.Ds.float().view(-1)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1)
#         out_y = self.selective_scan(
#             xs, dts,
#             As, Bs, Cs, Ds, z=None,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#             return_last_state=False,
#         ).view(B, K, -1, L)
#         assert out_y.dtype == torch.float

#         inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
#         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#         invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

#         return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

#     def forward(self, x: torch.Tensor, **kwargs):
#         B, C, H, W = x.shape
#         y1, y2, y3, y4 = self.forward_core(x)
#         assert y1.dtype == torch.float32
#         y = y1 + y2 + y3 + y4
#         y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
#         return y


# class MB(nn.Module):
#     def __init__(
#             self,
#             d_model,
#             d_state=16,
#             expand=2.,
#             dropout=0.,
#             bias=False,
#             device=None,
#             dtype=None,
#             **kwargs,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
#         self.out_norm1 = nn.LayerNorm(self.d_inner)
#         self.out_norm2 = nn.LayerNorm(self.d_inner)
#         self.out_norm3 = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else None
#         self.pooling = nn.MaxPool2d(kernel_size=(2, 2))
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.ca1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 groups=self.d_inner,
#                 bias=True,
#                 kernel_size=3,
#                 padding=(3 - 1) // 2, ),
#             nn.SiLU())
#         self.ca2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 groups=self.d_inner,
#                 bias=True,
#                 kernel_size=3,
#                 padding=(3 - 1) // 2, ),
#             nn.SiLU())
#         self.ca3 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 groups=self.d_inner,
#                 bias=True,
#                 kernel_size=3,
#                 padding=(3 - 1) // 2, ),
#             nn.SiLU())
#         self.ssm1 = SSM2D_MB(d_model=self.d_model, d_state=self.d_state, expand=self.expand,
#                              **kwargs)
#         self.ssm2 = SSM2D_MB(d_model=self.d_model, d_state=self.d_state, expand=self.expand,
#                              **kwargs)
#         self.ssm3 = SSM2D_MB(d_model=self.d_model, d_state=self.d_state, expand=self.expand,
#                              **kwargs)

#     def forward(self, x: torch.Tensor, **kwargs):
#         B, H, W, C = x.shape
#         skip = x
#         xz = self.in_proj(x)
#         x, z = xz.chunk(2, dim=-1)
#         x = x.permute(0, 3, 1, 2).contiguous()
#         x = self.ca1(x)
#         x2 = self.pooling(x)
#         y1 = self.ssm1(x)
#         y1 = self.out_norm1(y1)

#         x2 = self.ca2(x2)
#         x3 = self.pooling(x2)
#         y2 = self.ssm2(x2)
#         y2 = self.out_norm2(y2)

#         x3 = self.ca3(x3)
#         y3 = self.ssm3(x3)
#         y3 = self.out_norm3(y3)

#         y3 = y3.permute(0, 3, 1, 2).contiguous()
#         y3 = self.up(y3)
#         y2 = y2.permute(0, 3, 1, 2).contiguous()
#         y2 = y2 + y3
#         y2 = self.up(y2)
#         y2 = y2.permute(0, 2, 3, 1).contiguous()
#         y = y1 + y2

#         y = y * F.silu(z)
#         out = self.out_proj(y)
#         out = out + skip
#         if self.dropout is not None:
#             out = self.dropout(out)
#         return out

# class TIP(nn.Module):
#     def __init__(self, channel, drop_path: float = 0, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),):
#         super().__init__()
#         # self.ceb = CEB(channel)
#         self.feb = FEB(channel)
#         self.DAFM = DAFM(channel)
#         self.LN = nn.LayerNorm(channel)
#         self.ln = norm_layer(channel)
#         self.drop_path = DropPath(drop_path)
#     def forward(self, x):# 输入 BHWC 输出 BHWC
#         identity = x
#         x = x.permute(0, 2, 3, 1).contiguous()  # BCHW -> BHWC
#         x = self.ln(x)
#         # self.mamba = MB(d_model=hidden_dim, d_state=d_state, expand=mlp_ratio, dropout=attn_drop_rate,
#         #                 **kwargs)
#         out1 = self.drop_path(self.feb(x))
#         out1 = out1.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW
#         out2 = self.DAFM(identity)
#         out = out2 + out1
#         return out
    
class TIP(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.ceb = CEB(channel)
        self.feb = FEB(channel)
        # self.DAFM = DAFM(channel)
        self.LN = nn.LayerNorm(channel)
        # self.ln = norm_layer(channel)
        # self.drop_path = DropPath(drop_path)
    def forward(self, x):# 输入 BHWC 输出 BHWC
        x = x.permute(0, 2, 3, 1).contiguous()  # BCHW -> BHWC
        x = self.LN(x)
        # self.mamba = MB(d_model=hidden_dim, d_state=d_state, expand=mlp_ratio, dropout=attn_drop_rate,
        #                 **kwargs)
        out1 = self.ceb(x)
        out2 = self.feb(x)
        out = out2 + out1
        out = out.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW
        return out

class LayerNorm2d(nn.Module):
    """
    LayerNorm over channel dimension for 2D feature maps.
    x: (B, C, H, W)
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class ECALayer(nn.Module):
    """
    ECA-Net style efficient channel attention:
    GAP -> 1D conv over channels -> sigmoid
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        assert k_size % 2 == 1, "k_size should be odd"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.avg_pool(x)                 # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv1d(y)                   # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return self.sigmoid(y)


class LSKA(nn.Module):
    """
    Large Separable Kernel Attention (LSKA-like):
    depthwise (1xk -> kx1) + dilated depthwise (1xk -> kx1) + pointwise
    This keeps it efficient while enlarging receptive field.
    """
    def __init__(self, channels: int, kernel_size: int = 11, dilation: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        k = kernel_size
        pad = k // 2
        dpad = pad * dilation

        self.dw_1 = nn.Conv2d(channels, channels, kernel_size=(1, k),
                              padding=(0, pad), groups=channels, bias=False)
        self.dw_2 = nn.Conv2d(channels, channels, kernel_size=(k, 1),
                              padding=(pad, 0), groups=channels, bias=False)

        self.dw_3 = nn.Conv2d(channels, channels, kernel_size=(1, k),
                              padding=(0, dpad), dilation=(1, dilation),
                              groups=channels, bias=False)
        self.dw_4 = nn.Conv2d(channels, channels, kernel_size=(k, 1),
                              padding=(dpad, 0), dilation=(dilation, 1),
                              groups=channels, bias=False)

        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.dw_1(x)
        attn = self.dw_2(attn)
        attn = self.dw_3(attn)
        attn = self.dw_4(attn)
        attn = self.pw(attn)
        return x * attn


class GDFN(nn.Module):
    """
    Restormer-style Gated-Dconv FFN:
    1x1 -> DWConv -> split -> GELU gate -> 1x1
    """
    def __init__(self, dim: int, expansion: float = 2.0, dw_kernel: int = 3):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=dw_kernel,
                                padding=dw_kernel // 2, groups=hidden * 2, bias=True)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class CGLKM(nn.Module):
    """
    Context-Gated Large-Kernel Modulation block.
    Plug-and-play enhancement module for (B,C,H,W) features.
    """
    def __init__(
        self,
        channels: int,
        lsk_kernel: int = 11,
        lsk_dilation: int = 3,
        eca_kernel: int = 3,
        ffn_expansion: float = 2.0,
        norm: str = "ln2d",   # "ln2d" or "bn"
    ):
        super().__init__()
        if norm == "bn":
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
        else:
            self.norm1 = LayerNorm2d(channels)
            self.norm2 = LayerNorm2d(channels)

        self.spatial = LSKA(channels, kernel_size=lsk_kernel, dilation=lsk_dilation)
        self.channel_gate = ECALayer(channels, k_size=eca_kernel)

        # lightweight mixing after modulation
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.ffn = GDFN(channels, expansion=ffn_expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) large-kernel spatial context
        xs = self.spatial(self.norm1(x))              # (B,C,H,W)

        # 2) efficient channel gate (computed from original x for stability)
        gc = self.channel_gate(x)                     # (B,C,1,1)

        # 3) modulation + residual
        x = x + self.mix(xs * gc)

        # 4) gated feed-forward transform + residual
        x = x + self.ffn(self.norm2(x))
        return x