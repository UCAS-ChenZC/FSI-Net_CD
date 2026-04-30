import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial
from models.BaseNetworks import *
import torch.nn.functional as F
from mmengine.model.weight_init import normal_init
from models.pixel_shuffel_up import PS_UP
# from mmcv.utils import get_logger
from mmengine.logging import MMLogger
import logging
import warnings
# from mmcv.runner import load_checkpoint
from mmengine.runner import load_checkpoint as mmengine_load_checkpoint
from models.newFreqFusion import FreqFusion
from models.LIFT_IFF import FreqWaveletFusion
import torch.fft
import warnings
from models.FFSwin import dct_channel_block
from models.SAM import SAM
from models.Dysample import DySample_UP
from einops import rearrange

from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM
warnings.filterwarnings('ignore')

from timm.models.layers import trunc_normal_
from mmseg.models.utils import SelfAttentionBlock
# Transformer Decoder
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, dropout_rate=0.1):
        super(BiDirectionalCrossAttention, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        self.query_weight = nn.Parameter(torch.randn(input_dim, bottleneck_dim))
        self.key_weight = nn.Parameter(torch.randn(input_dim, bottleneck_dim))
        self.value_weight = nn.Parameter(torch.randn(input_dim, bottleneck_dim))
        self.dropout = nn.Dropout(dropout_rate)  # 添加 dropout 层

    def forward(self, Q, K, V):
        # 假设 Q, K, V 的形状均为 (batch_size, channels, height, width)
        batch_size, channels, height, width = Q.size()
        seq_len = height * width

        # 将 Q, K, V 展平为 (batch_size, seq_len, channels)
        Q_flat = Q.view(batch_size, seq_len, channels)
        K_flat = K.view(batch_size, seq_len, channels)
        V_flat = V.view(batch_size, seq_len, channels)

        # 缩放因子
        scale = torch.sqrt(torch.tensor(self.bottleneck_dim, dtype=torch.float32))

        # ----- 分支 A：以 Q 为查询，K 为键，V 为值 -----
        Q_A = torch.matmul(Q_flat, self.query_weight)
        K_A = torch.matmul(K_flat, self.key_weight)
        V_A = torch.matmul(V_flat, self.value_weight)
        scores_A = torch.matmul(Q_A, K_A.transpose(-2, -1)) / scale  # (batch_size, seq_len, seq_len)
        attn_A = F.softmax(scores_A, dim=-1)
        # attn_A = self.dropout(attn_A)  # 应用 dropout
        output_A = torch.matmul(attn_A, V_A)  # (batch_size, seq_len, bottleneck_dim)

        # ----- 分支 B：交换 Q 与 K（即以 K 为查询，Q 为键），依然使用 V 为值 -----
        Q_B = torch.matmul(K_flat, self.query_weight)
        K_B = torch.matmul(Q_flat, self.key_weight)
        V_B = torch.matmul(V_flat, self.value_weight)
        scores_B = torch.matmul(Q_B, K_B.transpose(-2, -1)) / scale
        attn_B = F.softmax(scores_B, dim=-1)
        # attn_B = self.dropout(attn_B)  # 应用 dropout
        output_B = torch.matmul(attn_B, V_B)

        # 将两个分支的输出相加
        output = output_A + output_B  # (batch_size, seq_len, bottleneck_dim)

        # 恢复为 (batch_size, bottleneck_dim, height, width)
        output = output.transpose(1, 2).view(batch_size, self.bottleneck_dim, height, width)
        return output



#可迁移至第二篇
class MutualCrossAttention(nn.Module):
    def __init__(self, dropout):
        super(MutualCrossAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        # 输入形状: (2, 256, 8, 8)
        batch_size, channels, h, w = x1.shape
        
        # 将空间维度展平为序列维度 [2, 256, 8, 8] → [2, 256, 64]
        x1_flat = x1.view(batch_size, channels, -1).transpose(1, 2)  # [2, 64, 256]
        x2_flat = x2.view(batch_size, channels, -1).transpose(1, 2)  # [2, 64, 256]

        # 计算注意力分数
        d = x1_flat.shape[-1]  # d = 256
        scores = torch.bmm(x1_flat, x2_flat.transpose(1, 2)) / math.sqrt(d)  # [2,64,64]
        output_A = torch.bmm(self.dropout(F.softmax(scores, dim=-1)), x2_flat)  # [2,64,256]

        scores = torch.bmm(x2_flat, x1_flat.transpose(1, 2)) / math.sqrt(d)  # [2,64,64]
        output_B = torch.bmm(self.dropout(F.softmax(scores, dim=-1)), x1_flat)  # [2,64,256]

        # 合并输出并恢复空间维度
        output = (output_A + output_B).transpose(1, 2)  # [2,256,64]
        output = output.view(batch_size, channels, h, w)  # [2,256,8,8]

        return output

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),     #bias= False是后加的
    #添加：BN RELU
        nn.BatchNorm2d(out_channels),  # 先进行 BN 归一化
        nn.LeakyReLU(negative_slope=0.1, inplace=True),  # LeakyReLU 代替 ReLU

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    #添加：Dropuot
        nn.Dropout(0.3)  # 增加 Dropout，减少过拟合
    )

def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),  # 先进行 BN 归一化
        nn.LeakyReLU(negative_slope=0.1, inplace=True),  # LeakyReLU 代替 ReLU
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Dropout(0.3)  # 增加 Dropout，减少过拟合
    )
class DiffAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiffAttentionModule, self).__init__()
        
        # 通道注意力机制
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),  # 恢复
            nn.Sigmoid()
        )
        self.BR = nn.Sequential(
                                ResidualBlock(out_channels),
                                ResidualBlock(out_channels),
                                ResidualBlock(out_channels),
                                nn.ReLU()  
        )
        
        # 1x1卷积扩展通道
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv3x3 = nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(num_features=out_channels),
        #             nn.ReLU()
        # )
        self.layer_norm = nn.LayerNorm(out_channels)  # 层归一化
    def forward(self, F_A, F_B):
        # 计算特征差异
        F_diff = torch.abs(F_A - F_B)  # 形状: (B, C, H, W)
        
        # 计算全局通道注意力
        gap = self.global_avg_pool(F_diff).view(F_diff.shape[0], -1)  # (B, C)
        attn_weights = self.mlp(gap).view(F_diff.shape[0], F_diff.shape[1], 1, 1)  # (B, C, 1, 1)
        F_weighted = F_diff * attn_weights  # 加权
        F_weighted = self.conv1x1(F_weighted)
        F_res = self.BR(F_weighted)
        F_out = F_weighted + F_res
        # 1x1卷积扩展通道
        # F_out = self.conv1x1(F_out)  # (B, 256, H, W)
        # F_out = self.conv3x3(F_weighted)  # (B, 256, H, W)
        x = F_out.permute(0, 2, 3, 1)
        x = self.layer_norm(x)  # 层归一化
        x = x.permute(0, 3, 1, 2)
        return x
    
class SegFusion(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3,4], align_corners=True, 
                    in_channels = [48, 96, 192, 384], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16,32]):
        super(SegFusion, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c0_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c0   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.embedding_dim)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.embedding_dim)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.embedding_dim)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.embedding_dim)
        self.make_pred_c0 = make_prediction(in_channels=self.embedding_dim, out_channels=self.embedding_dim)
        self.make_pred_final = make_prediction(in_channels=self.embedding_dim, out_channels=self.embedding_dim)

        self.pred_final_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.pred_final_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.pred_final_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.pred_final_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.pred_final_c0 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        #Final linear fusion layer
        # self.linear_fuse = nn.Sequential(
        #     nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
        #                                 kernel_size=1),
        #     nn.BatchNorm2d(self.embedding_dim)
        # )
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*14, out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 
        # self.Conv_cc1 = nn.Conv2d(512,256,kernel_size=1)
        # self.Conv_cc2 = nn.Conv2d(768,256,kernel_size=1)
        # self.Conv_cc3 = nn.Conv2d(1024,256,kernel_size=1)
        self.Conv_cc4 = nn.Conv2d(1280,256,kernel_size=1)   #1280
        # self.CRA = ImprovedCRA(256,reduction_ratio=4)
        self.ff1 = FreqFusion(256, 256)
        self.ff2 = FreqFusion(256,512)
        self.ff3 = FreqFusion(256,768)
        self.ff4 = FreqFusion(256,1024)
        # self.ff1 = FreqWaveletFusion(256, 256)
        # self.ff2 = FreqWaveletFusion(256,512)
        # self.ff3 = FreqWaveletFusion(256,768)
        # self.ff4 = FreqWaveletFusion(256,1024)
        self.Dysample = DySample_UP(in_channels=256,scale=2,style='pl')
        # self.Dysample_cc1 = DySample_UP(in_channels=512,scale=8,style='pl')
        # self.Dysample_cc2 = DySample_UP(in_channels=768,scale=4,style='pl')
        # self.Dysample_cc3 = DySample_UP(in_channels=1024,scale=2,style='pl')
        self.DAFM =DAFM(256,input_size=(256,256))
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2) 

    def _transform_inputs(self, inputs):                                        #对输入特征进行转换
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs
    def to_3d(self,x):
        return rearrange(x, 'b c h w -> b (h w) c')
    def reshape_to_4d(self,x):
        batch_size, num_patches, channels = x.shape
        H = W = int(num_patches ** 0.5)
        return x.reshape(batch_size, channels, H, W)
    def forward(self, inputs1, inputs2):
        '''处理输入数据'''
    #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16
       # x_1 = inputs1  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
       # x_2 = inputs2  # len=4, 1/2, 1/4, 1/8, 1/16
    #img1 and img2 features
        c0_1,c1_1, c2_1, c3_1, c4_1 = x_1
        c0_2,c1_2, c2_2, c3_2, c4_2 = x_2
        # c1_1, c2_1, c3_1, c4_1 = x_1
        # c1_2, c2_2, c3_2, c4_2 = x_2        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
    # Stage 4: x1/32 scale (8,8)
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        # _c4_MCA = self.BMCA(_c4_1,_c4_2)
        _c4 = torch.cat((_c4_1, _c4_2), dim=1)
        _c4   = self.diff_c4(_c4)
        # _c4   = self.DAttention(_c4)

        #_c4 = self.DiffAttentionModule4(_c4_1,_c4_2)
        output_c4 = self.make_pred_c4(_c4)
        pre_4 = self.pred_final_c4(_c4)
        outputs.append(pre_4)
        # _c4 = self.DAFM(_c4)

    # Stage 3: x1/16 scale (16,16)
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        # _c3_MCA = self.BMCA(_c3_1,_c3_2)
        _c3 = torch.cat((_c3_1, _c3_2), dim=1)
        _c3   = self.diff_c3(_c3)
        # _c3   = self.DAttention(_c3)

        #_c3 = self.DiffAttentionModule3(_c3_1,_c3_2)
        output_c3 = self.make_pred_c3(_c3)
        pre_3 = self.pred_final_c3(_c3)
        outputs.append(pre_3)
        # _c3 = self.DAFM(_c3)

    # Stage 2: x1/8 scale (32,32)
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        # _c2_MCA = self.BMCA(_c2_1,_c2_2)
        _c2 = torch.cat((_c2_1, _c2_2), dim=1)
        # _c2 = self.MCA(_c2_1,_c2_2)
        _c2   = self.diff_c2(_c2)
        # _c2   = self.DAttention(_c2)

        #_c2 = self.DiffAttentionModule2(_c2_1,_c2_2)
        output_c2 = self.make_pred_c2(_c2)
        pre_2 = self.pred_final_c2(_c2)
        outputs.append(pre_2)
        # _c2 = self.DAFM(_c2)

    # Stage 1: x1/4 scale (64,64)
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        # _c1_MCA = self.BMCA(_c1_1,_c1_2)
        _c1 = torch.cat((_c1_1, _c1_2), dim=1)
        # _c1 = self.MCA(_c1_1,_c1_2)
        _c1   = self.diff_c1(_c1)

        #_c1 = self.DiffAttentionModule1(_c1_1,_c1_2)
        output_c1 = self.make_pred_c1(_c1)
        pre_1 = self.pred_final_c1(_c1)
        outputs.append(pre_1)
        # _c1 = self.DAFM(_c1)
       
    # Stage 0: X1/2 scale (128,128)
        _c0_1 = self.linear_c0(c0_1).permute(0,2,1).reshape(n, -1, c0_1.shape[2], c0_1.shape[3])
        _c0_2 = self.linear_c0(c0_2).permute(0,2,1).reshape(n, -1, c0_2.shape[2], c0_2.shape[3])
        # _c0_MCA = self.BMCA(_c0_1,_c0_2)
        _c0 = torch.cat((_c0_1, _c0_2), dim=1)
        # _c0 = self.MCA(_c0_1,_c0_2)
        _c0   = self.diff_c0(_c0)

        # _c0 = self.DiffAttentionModule0(_c0_1,_c0_2)
        output_c0 = self.make_pred_c0(_c0)
        # pre_0 = self.pred_final_c0(_c0)
        # outputs.append(pre_0)
        # _c0 = self.DAFM(_c0)

    #FreqFusion模块
        _, x3, x4_up = self.ff1(hr_feat=output_c3, lr_feat=output_c4)
        cc1 = torch.cat([x3, x4_up],dim=1)
        _, x2, x34_up = self.ff2(hr_feat=output_c2, lr_feat= cc1)
        cc2 = torch.cat([x2, x34_up],dim=1)
        _, x1, x234_up = self.ff3(hr_feat=output_c1, lr_feat=cc2)
        cc3 = torch.cat([x1, x234_up],dim=1) # channel=4c, 1/4 img size
        # cc3 = self.Conv_cc3(cc3)            #c 1024 -> 256 
        # cc3 = x = self.convd2x(cc3)
        # cc = cc3*_c0
        _, x0, x1234_up = self.ff4(hr_feat=output_c0, lr_feat=cc3)
        cc4 = torch.cat([x0, x1234_up],dim=1)

        # x4_up = self.ff1(hr_feat=output_c3, lr_feat=output_c4)
        # # cc1 = torch.cat([output_c3, x4_up],dim=1)
        # x34_up = self.ff2(hr_feat=output_c2, lr_feat= x4_up)
        # # cc2 = torch.cat([output_c2, x34_up],dim=1)
        # x234_up = self.ff3(hr_feat=output_c1, lr_feat=x34_up)
        # # cc3 = torch.cat([output_c1, x234_up],dim=1) # channel=4c, 1/4 img size
        # # cc3 = self.Conv_cc3(cc3)            #c 1024 -> 256 
        # # cc3 = x = self.convd2x(cc3)
        # # cc = cc3*_c0
        # x1234_up = self.ff4(hr_feat=output_c0, lr_feat=x234_up)
        # # cc4 = torch.cat([output_c0, x1234_up],dim=1)
    #尝试在多尺度频域特征融合后进行  通道、尺寸的对齐相加
        # cc1 = self.Dysample_cc1(cc1)
        # cc1 = self.Conv_cc1(cc1)            #c 512  -> 256 
        # cc2 = self.Dysample_cc2(cc2)
        # cc2 = self.Conv_cc2(cc2)            #c 768  -> 256 
        # cc3 = self.Dysample_cc3(cc3)
        # # cc3 = self.Conv_cc3(cc3)            #c 1024 -> 256 

        # _c = self.linear_fuse(torch.cat((cc1, cc2, cc3, cc4), dim=1))

        cc4 = self.Conv_cc4(cc4)            #c 1280 -> 256    1*1卷积
        # x = self.DAFM(cc4)
        # x = _c + cc4
        x = self.make_pred_final(cc4)       #c 1280 -> 256      pred模块
        x = self.DAFM(x)
        # out_cls_mid, cls_tokens = self.class_token(x)      #size _c    (b,256,64,64)
        # out_new = self.trans(x, cls_tokens, out_cls_mid)
        # x=self.pooling(x)
        #x = self.DAttention(cc4)
        # x = self.pooling(x)

        # _c = cc1 + cc2 + cc3 +cc4         #nn.BatchNorm2d(self.embedding_dim)
        # _c = self.DynamicChannelReducer(_c)


        #Linear Fusion of difference image from all scales
        # _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1,_c0_up), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

    #     Upsampling x2 (x1/2 scale)
        # x = self.convd2x(out_new)
        # # #Residual block
        # x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        # x = self.convd1x(_c)
        x = self.Dysample(x)
        # #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs
    



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
        self.dilated_conv_1 = nn.Sequential(
                    nn.Conv2d(channel, channel,kernel_size=3, padding=2, dilation=2),
                    nn.Conv2d(channel, channel,kernel_size=3, padding=4, dilation=4),
                    nn.BatchNorm2d(num_features=channel),
                    nn.ReLU())
        self.dilated_conv_2 = nn.Sequential(
                    nn.Conv2d(channel * 2, channel * 2,kernel_size=3, padding=2, dilation=2),
                    nn.Conv2d(channel * 2, channel,kernel_size=3, padding=4, dilation=4),
                    nn.BatchNorm2d(num_features=channel),
                    nn.ReLU())
        self.dilated_conv_3 = nn.Sequential(
                    nn.Conv2d(channel, channel,kernel_size=3, padding=2, dilation=2),
                    nn.Conv2d(channel, channel,kernel_size=3, padding=4, dilation=4),
                    nn.BatchNorm2d(num_features=channel),
                    nn.ReLU())
        self.Conv1 = nn.Conv2d(channel * 2,channel * 2,kernel_size=1)
        H,W = input_size
        # self.FAM = FAM_Module(channel,channel,shapes=H)
    def forward(self,x):
        
        x1 = x
        fc = x1 * self.channelattention(x1)     #Wc = self.channelattention(x1)
        fc = self.dilated_conv_1(fc)              #   dilated_conv    Conv3_1     FAM
        fs = fc * self.spatialattention(fc)     #Ws = self.spatialattention(fc)
        cat = torch.concat([x1,fs],dim=1)
        fd = cat * self.Conv1(cat)
        fd = self.Conv3_2(fd)
        # f_out_res = self.Conv3_3(f_out)
        f_out_res = self.Conv3_3(fd)
        f_out = fd + f_out_res
        x  = x + f_out
        return x
    


class Decoder_v4_FpnFusion(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3,4], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 256, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16,32]):
        super(Decoder_v4_FpnFusion, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c0_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c0   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c0 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 
        self.Convd1_4 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_3 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_2 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_1 = nn.Conv2d(1280,256,kernel_size=1)
        self.Conv = nn.Conv2d(512,256,kernel_size=1)
        self.ff1 = FreqFusion(256, 256)
        self.ff2 = FreqFusion(256,256)
        self.ff3 = FreqFusion(256,256)
        self.ff4 = FreqFusion(256,256)

    def _transform_inputs(self, inputs):                                        #对输入特征进行转换
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs1, inputs2):
        '''处理输入数据'''
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c0_1,c1_1, c2_1, c3_1, c4_1 = x_1
        c0_2,c1_2, c2_2, c3_2, c4_2 = x_2
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1))

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1))

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1))

       # Stage 0: x1 scale
        _c0_1 = self.linear_c0(c0_1).permute(0,2,1).reshape(n, -1, c0_1.shape[2], c0_1.shape[3])
        _c0_2 = self.linear_c0(c0_2).permute(0,2,1).reshape(n, -1, c0_2.shape[2], c0_2.shape[3])
        _c0 = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1))

        #FreqFusion模块__UNetFusion
        _, x3, x4_up = self.ff1(hr_feat=_c3, lr_feat=_c4)
        cc1 = x3 + x4_up
        _, x2, x34_up = self.ff2(hr_feat=_c2, lr_feat= cc1)
        cc2 = x2 + x34_up
        _, x1, x234_up = self.ff3(hr_feat=_c1, lr_feat=cc2)
        cc3 = x1 + x234_up
        _, x0, x1234_up = self.ff4(hr_feat=_c0, lr_feat=cc3)
        _c = x0 + x1234_up                                                      # channel=c, 1/2 img size
        #_c = self.Convd1_1(_c)


        #Upsampling x2 (x1/2 scale)
        # x = self.convd2x(_c)
        # #Residual block
        # x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(_c)
        # #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs
    

class Decoder_v4_UNetFusion(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3,4], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 256, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16,32]):
        super(Decoder_v4_UNetFusion, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c0_in_channels,c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c0   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c0 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 
        self.Convd1_4 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_3 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_2 = nn.Conv2d(768,256,kernel_size=1)
        self.Convd1_1 = nn.Conv2d(1280,256,kernel_size=1)
        self.Conv = nn.Conv2d(512,256,kernel_size=1)
        self.ff1 = FreqFusion(256, 256)
        self.ff2 = FreqFusion(256,256)
        self.ff3 = FreqFusion(256,256)
        self.ff4 = FreqFusion(256,256)

    def _transform_inputs(self, inputs):                                        #对输入特征进行转换
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs1, inputs2):
        '''处理输入数据'''
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c0_1,c1_1, c2_1, c3_1, c4_1 = x_1
        c0_2,c1_2, c2_2, c3_2, c4_2 = x_2
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1))

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1))

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1))

       # Stage 0: x1 scale
        _c0_1 = self.linear_c0(c0_1).permute(0,2,1).reshape(n, -1, c0_1.shape[2], c0_1.shape[3])
        _c0_2 = self.linear_c0(c0_2).permute(0,2,1).reshape(n, -1, c0_2.shape[2], c0_2.shape[3])
        _c0 = self.diff_c0(torch.cat((_c0_1, _c0_2), dim=1))

        #FreqFusion模块__UNetFusion
        _, x3, x4_up = self.ff1(hr_feat=_c3, lr_feat=_c4)
        cc1 = self.Conv(torch.cat([x3, x4_up],dim=1))
        _, x2, x34_up = self.ff2(hr_feat=_c2, lr_feat= cc1)
        cc2 = self.Conv(torch.cat([x2, x34_up],dim=1))
        _, x1, x234_up = self.ff3(hr_feat=_c1, lr_feat=cc2)
        cc3 = self.Conv(torch.cat([x1, x234_up],dim=1))
        _, x0, x1234_up = self.ff4(hr_feat=_c0, lr_feat=cc3)                        # channel=c, 1/2 img size
        _c = self.Conv(torch.cat([x0, x1234_up],dim=1))
        #_c = self.Convd1_1(_c)

        #Linear Fusion of difference image from all scales
        #_c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1,_c0_up), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        #Upsampling x2 (x1/2 scale)
        # x = self.convd2x(_c)
        # #Residual block
        # x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(_c)
        # #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs
    

class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """
    # def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3,4], align_corners=True, 
    #                 in_channels = [32, 64, 128, 256], embedding_dim= 256, output_nc=2, 
    #                 decoder_softmax = False, feature_strides=[2, 4, 8, 16,32]):
    #     super(Decoder_v4_UNetFusion, self).__init__()
    def __init__(self, input_transform='multiple_select',in_index=[0, 1, 2, 3],align_corners=False,
                    in_channels=[96, 192, 368, 784],channels=512,dropout_ratio=0.1,num_classes=2,pool_scales=(1, 2, 3, 6),conv_cfg=None,norm_cfg=None,act_cfg = dict(type='ReLU')):
        super(UPerHead,self).__init__()
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.channels = channels
        self.num_classes = num_classes
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.conv_seg = nn.Conv2d(2*channels, num_classes, kernel_size=1)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.convd1x = UpsampleConvLayer(2,2, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(channels))
        self.change_probability = ConvLayer(channels,self.num_classes, kernel_size=3, stride=1, padding=1)
        self.active  = nn.Sigmoid() 
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
    
    def _transform_inputs(self, inputs):                                        #对输入特征进行转换
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    # def forward(self, inputs1, inputs2):
    #     '''处理输入数据'''
    #     #Transforming encoder features (select layers)
    #     x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
    #     x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16
    def _forward_feature(self, CC):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # inputs = self._transform_inputs(inputs)
        # x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16                 (16 64 64 64) --> (16 64 64 64)
        # x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16
        # build laterals
        outputs = []
        for i,tt in enumerate(CC):
            inputs = tt
            inputs = self._transform_inputs(inputs)
            laterals = [
                lateral_conv(inputs[i])
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]

            laterals.append(self.psp_forward(inputs))

            # build top-down path
            used_backbone_levels = len(laterals)
            for i in range(used_backbone_levels - 1, 0, -1):
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + resize(
                    laterals[i],
                    size=prev_shape,
                    mode='bilinear',
                    align_corners=self.align_corners)

            # build outputs
            fpn_outs = [
                self.fpn_convs[i](laterals[i])
                for i in range(used_backbone_levels - 1)
            ]
            # append psp feature
            fpn_outs.append(laterals[-1])

            for i in range(used_backbone_levels - 1, 0, -1):
                fpn_outs[i] = resize(
                    fpn_outs[i],
                    size=fpn_outs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
            fpn_outs = torch.cat(fpn_outs, dim=1)
            feats = self.fpn_bottleneck(fpn_outs)
            outputs.append(feats)
        return outputs

    def forward(self, CC):
        """Forward function."""
        out = []
        feature_output = self._forward_feature(CC)
        output = torch.cat((feature_output[0],feature_output[1]), dim=1)
        finally_output = self.cls_seg(output)
        finally_output = self.convd1x(finally_output)          #(2 2 128 128)
        # finally_output = self.dense_1x(finally_output)
        # finally_output = self.change_probability(finally_output)
        # finally_output = self.active(finally_output)
        out.append(finally_output)
        return out
   