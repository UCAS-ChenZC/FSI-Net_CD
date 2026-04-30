import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath, trunc_normal_
from typing import List
# from ..builder import BACKBONES
# from mmengine.runner import load_checkpoint as mmengine_load_checkpoint
from mmengine.runner import load_checkpoint
# from mmseg.utils import get_root_logger
from typing import Tuple, Dict
import sys
import os
import logging
from einops import rearrange
# from models.DynamicFilter import DynamicFilter
from models.DynamicFilter_117 import DynamicFilter

logger = logging.getLogger()
def get_root_logger():
    return logger


# class RealFFT2D(nn.Module):
#     def forward(self, x):
#         # 输入形状: (B, C, H, W)
#         x_fft = torch.fft.rfft2(x, norm='ortho')
#         # 拆分为实部和虚部
#         x = torch.cat([x_fft.real, x_fft.imag], dim=1)  # 输出形状: (B, 2C, H, W//2+1)
#         return x

# class InvRealFFT2D(nn.Module):
#     def forward(self, x, original_shape):
#         # 输入形状: (B, 2C, H, W_rfft)
#         C = x.size(1) // 2
#         H, W = original_shape
        
#         # 合并实部和虚部
#         x_complex = torch.complex(x[:, :C, :, :], x[:, C:, :, :])
#         # 逆FFT恢复空间域
#         x = torch.fft.irfft2(x_complex, s=(H, W), norm='ortho')
#         return x
# class FFM(nn.Module):
#     def __init__(self, channels=96):
#         super().__init__()
        
#         # 左分支（空间域）
#         self.left_path = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(channels, channels, 3, padding=1)
#         )
        
#         # 右分支（频域）
#         self.right_Conv= nn.Sequential(
#             nn.Conv2d(channels, channels, 1),            #应该是1*1卷积
#             nn.LeakyReLU()
#             )
#         self.right_path = nn.Sequential(
#             RealFFT2D(),
#             nn.Conv2d(channels*2, channels*2, 1),        #应该是1*1卷积
#             nn.LeakyReLU())
#         self.inv_fft = InvRealFFT2D()
        
#         # 合并后的处理
#         self.merge_conv_left = nn.Conv2d(channels, channels, 1)
#         self.merge_conv_right = nn.Conv2d(channels, channels, 1)
#         self.final_conv = nn.Sequential(
#             nn.Conv2d(channels*2, channels, 1)
#         )

#     def forward(self, x):
#         # 原始空间尺寸
#         original_shape = x.shape[2:]
        
#         # 左分支处理
#         left = self.left_path(x)
#         left_output = x + left
        
#         # 右分支处理
#         right_1 = self.right_Conv(x)
#         right_2 = self.right_path(right_1)
#         right_res = right_1 + self.inv_fft(right_2,original_shape)
#         right_output = self.merge_conv_right(right_res)
#         # right_output = self.inv_fft(right, original_shape)
        
#         # 分支合并
#         # left = self.merge_conv_left(left_output)
        
        
#         # 最终合并和输出
#         out = torch.cat([left_output, right_output], dim=1)
#         return self.final_conv(out)
    
class SEGate(nn.Module):
    """Squeeze-Excitation gate: returns (B,C,1,1) in (0,1)."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.mean(dim=(2, 3), keepdim=True)
        g = self.fc2(self.act(self.fc1(g)))
        return torch.sigmoid(g)


class SFIBv2(nn.Module):
    """
    Phase-aware Radial-Band Spectral Interaction Block (SFIBv2)

    Key ideas:
      - Radial band decomposition on rFFT spectrum
      - Dynamic amplitude + phase modulation per band (conditioned on input)
      - Bidirectional cross-gating between spatial and frequency branches
    """
    def __init__(
        self,
        channels: int,
        num_bands: int = 6,
        mlp_ratio: float = 0.25,
        se_reduction: int = 8,
        use_residual: bool = True,
    ):
        super().__init__()
        assert num_bands >= 3, "num_bands建议 >=3（低/中/高频）"
        self.channels = channels
        self.num_bands = num_bands
        self.use_residual = use_residual

        # -------- Spatial branch (local detail) --------
        self.spa_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.spa_pw = nn.Conv2d(channels, channels, 1, bias=True)
        self.spa_act = nn.SiLU(inplace=True)

        # -------- Frequency branch (global mixing) --------
        self.pre_freq = nn.Conv2d(channels, channels, 1, bias=True)

        # Complex mixer in frequency domain: conv on [real, imag]
        self.freq_mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 1, bias=True),
        )

        # Band-wise (amp, phase) predictor from pooled spatial features
        hidden = max(int(channels * mlp_ratio), 16)
        self.band_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * num_bands, bias=True),  # [amp_logits(K), phase_logits(K)]
        )

        # Learnable band bias (helps stability + acts like base spectral prior)
        self.band_amp_bias = nn.Parameter(torch.zeros(num_bands))
        self.band_phase_bias = nn.Parameter(torch.zeros(num_bands))

        # -------- Cross-domain gating --------
        self.gate_f2s = SEGate(channels, reduction=se_reduction)  # from freq -> gate spatial
        self.gate_s2f = SEGate(channels, reduction=se_reduction)  # from spatial -> gate freq

        # -------- Fusion --------
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )

        # Cache for radial masks per (H, W, device, dtype)
        self._mask_cache: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

    @torch.no_grad()
    def _build_radial_masks(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build K radial-band masks for rFFT2 output size (H, Wf),
        returned shape: (K, H, Wf), float tensor in {0,1}.
        """
        Wf = W // 2 + 1

        fy = torch.fft.fftfreq(H, d=1.0, device=device)  # (H,)
        fx = torch.fft.rfftfreq(W, d=1.0, device=device)  # (Wf,)
        fy = fy.view(H, 1).abs()
        fx = fx.view(1, Wf).abs()

        # radius in [0, 0.5*sqrt(2)] roughly (normalized)
        radius = torch.sqrt(fy * fy + fx * fx)  # (H, Wf)
        rmax = radius.max().clamp_min(1e-6)

        # Uniform bins on [0, rmax]
        edges = torch.linspace(0.0, float(rmax), steps=self.num_bands + 1, device=device)
        # bucket index in [0, K-1]
        band_idx = torch.bucketize(radius, edges[1:-1], right=False)  # (H, Wf)

        masks = []
        for k in range(self.num_bands):
            masks.append((band_idx == k).to(dtype))
        masks = torch.stack(masks, dim=0)  # (K, H, Wf)
        return masks

    def _get_masks(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (H, W, device, dtype)
        if key not in self._mask_cache:
            self._mask_cache[key] = self._build_radial_masks(H, W, device, dtype)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.channels, f"channels mismatch: got {C}, expect {self.channels}"

        # -------- Spatial branch --------
        spa = self.spa_pw(self.spa_act(self.spa_dw(x)))
        spa = spa + x  # local residual

        # -------- Frequency branch --------
        # For numeric stability (esp. AMP), do FFT in float32 then cast back
        x_dtype = x.dtype
        xf = self.pre_freq(x)
        xf32 = xf.float() if xf.dtype in (torch.float16, torch.bfloat16) else xf

        X = torch.fft.rfft2(xf32, norm="ortho")  # (B, C, H, Wf) complex
        Wf = W // 2 + 1

        # Predict band-wise amplitude and phase from pooled spatial features
        pooled = x.mean(dim=(2, 3))  # (B, C)
        logits = self.band_mlp(pooled)  # (B, 2K)
        amp_logits, phase_logits = logits[:, : self.num_bands], logits[:, self.num_bands :]

        # amplitude in (0, 2), phase in (-pi, pi)
        amp = 2.0 * torch.sigmoid(amp_logits + self.band_amp_bias.view(1, -1))  # (B, K)
        phase = math.pi * torch.tanh(phase_logits + self.band_phase_bias.view(1, -1))  # (B, K)

        masks = self._get_masks(H, W, device=x.device, dtype=xf32.dtype)  # (K, H, Wf)

        # Build per-position amplitude/phase maps by band masks
        # (B, H, Wf)
        amp_map = torch.einsum("bk,khw->bhw", amp, masks)
        phase_map = torch.einsum("bk,khw->bhw", phase, masks)

        # Complex factor: amp * exp(j*phase)
        factor = torch.polar(amp_map, phase_map).unsqueeze(1)  # (B, 1, H, Wf) complex
        X = X * factor  # broadcast to channels

        # Complex mixing via 1x1 conv on [real, imag]
        feat = torch.cat([X.real, X.imag], dim=1)  # (B, 2C, H, Wf)
        feat = self.freq_mix(feat)
        real, imag = feat.chunk(2, dim=1)
        X = torch.complex(real, imag)

        freq = torch.fft.irfft2(X, s=(H, W), norm="ortho")  # (B, C, H, W) real
        freq = freq.to(x_dtype)

        # -------- Explicit cross-domain interaction (gating) --------
        # freq -> gate spatial
        g_f2s = self.gate_f2s(freq)
        spa = spa * (1.0 + g_f2s)

        # spatial -> gate freq
        g_s2f = self.gate_s2f(spa)
        freq = freq * (1.0 + g_s2f)

        # -------- Fusion + residual --------
        out = self.fuse(torch.cat([spa, freq], dim=1))
        if self.use_residual:
            out = out + x
        return out
    
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        x = x.permute(0, 2, 3, 1) #(b h w c)
        return x
    

class RetNetRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2)
        mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None] #(n H*W H*W)
        return mask
    
    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :] #(l l)
        mask = mask.abs() #(l l)
        mask = mask * self.decay[:, None, None] #(n l l)
        return mask
    
    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0]*slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0]*slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            index = torch.arange(slen[0]*slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
            sin = sin.reshape(slen[0], slen[1], -1) #(h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
            cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)

            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(slen[0]*slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
            sin = sin.reshape(slen[0], slen[1], -1) #(h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
            cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)
            mask = self.generate_2d_decay(slen[0], slen[1]) #(n l l)
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos
    
class VisionRetentionChunk(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)


        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''
        
        qr_w = qr.transpose(1, 2) #(b h n w d1)
        kr_w = kr.transpose(1, 2) #(b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4) #(b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2) #(b h n w w)
        qk_mat_w = qk_mat_w + mask_w  #(b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1) #(b h n w w)
        v = torch.matmul(qk_mat_w, v) #(b h n w d2)


        qr_h = qr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        v = v.permute(0, 3, 2, 1, 4) #(b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2) #(b w n h h)
        qk_mat_h = qk_mat_h + mask_h  #(b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1) #(b w n h h)
        output = torch.matmul(qk_mat_h, v) #(b w n h d2)
        
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output
    
class VisionRetentionAll(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        
        assert h*w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        qr = theta_shift(q, sin, cos) #(b n h w d1)
        kr = theta_shift(k, sin, cos) #(b n h w d1)

        qr = qr.flatten(2, 3) #(b n l d1)
        kr = kr.flatten(2, 3) #(b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d2)
        vr = vr.flatten(2, 3) #(b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2) #(b n l l)
        qk_mat = qk_mat + mask  #(b n l l)
        qk_mat = torch.softmax(qk_mat, -1) #(b n l l)
        output = torch.matmul(qk_mat, vr) #(b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
    
class RetBlock(nn.Module):

    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = VisionRetentionChunk(embed_dim, num_heads)
        else:
            self.retention = VisionRetentionAll(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)

    def forward(
            self,
            x: torch.Tensor, 
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
        ):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.SyncBatchNorm(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = x.permute(0, 3, 1, 2).contiguous()  #(b c h w)
        x = self.reduction(x) #(b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) #(b oh ow oc)
        return x
    
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 init_value: float, heads_range: float,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
                 downsample: PatchMerging=None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        if chunkwise_recurrent:
            flag = 'chunk'
        else:
            flag = 'whole'
        self.Relpos = RetNetRelPos2d(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.blocks = nn.ModuleList([
            RetBlock(flag, embed_dim, num_heads, ffn_dim, 
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        self.SFIB_1 = SFIBv2(96)
        self.SFIB_2 = SFIBv2(192)
        self.SFIB_3 = SFIBv2(384)
        self.SFIB_4 = SFIBv2(768)
        # self.SFB1 = FFM(96)
        # self.SFB2 = FFM(192)
        # self.SFB3 = FFM(384)
        # self.SFB4 = FFM(768)

    def forward(self, x):
        b, h, w, d = x.size()
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        
        identity = x.permute(0, 3, 1, 2)
        if d == 96:
            query = self.SFIB_1(identity)
        elif d == 192:
            query = self.SFIB_2(identity)
        elif d == 384:
            query = self.SFIB_3(identity)
        elif d == 768:
            query = self.SFIB_4(identity)
        query = query.permute(0, 2, 3, 1)
        
        # identity = x.permute(0, 3, 1, 2)
        # if d == 96:
        #     query = self.SFB1(identity)
        # elif d == 192:
        #     query = self.SFB2(identity)
        # elif d == 384:
        #     query = self.SFB3(identity)
        # elif d == 768:
        #     query = self.SFB4(identity)
        # query = query.permute(0, 2, 3, 1)
        
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x=x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
            else:
                x = blk(x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
        # identity = x.permute(0, 3, 1, 2)
        # if d == 96:
        #     query = self.SFB1(identity)
        # elif d == 192:
        #     query = self.SFB2(identity)
        # elif d == 384:
        #     query = self.SFB3(identity)
        # elif d == 768:
        #     query = self.SFB4(identity)
        # query = query.permute(0, 2, 3, 1)
        
        x = x + query
        
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x
    
class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1).contiguous() #(b h w c)
        x = self.norm(x) #(b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1) #(b h w c)
        return x

# @BACKBONES.register_module()
class FSIFormer(nn.Module):

    def __init__(self, in_chans=32, out_indices=(0, 1, 2, 3),
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 init_values=[1, 1, 1, 1], heads_ranges=[3, 3, 3, 3], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 patch_norm=True, use_checkpoint=False, chunkwise_recurrents=[True, True, False, False], projection=1024,
                 layerscales=[False, False, False, False], layer_init_values=1e-6, norm_eval=True,):
        super().__init__()
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.norm_eval = norm_eval

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None)


        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values
            )
            self.layers.append(layer)

        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(nn.LayerNorm(embed_dims[i]))

        self.apply(self._init_weights)
    #2025.0218
        self.DynamicFilter_1 = DynamicFilter(96,size=64)
        self.DynamicFilter_2 = DynamicFilter(192,size=32)
        self.DynamicFilter_3 = DynamicFilter(384,size=16)
        self.DynamicFilter_4 = DynamicFilter(768,size=8)
        # self.df_norms = nn.ModuleList([nn.LayerNorm(d) for d in embed_dims])
        # self.df_drop = DropPath(drop_path_rate)  # 或者单独设一个 df_drop_rate
        # # 可选：layerscale
        # self.df_gamma = nn.ParameterList([nn.Parameter(1e-5*torch.ones(1,1,1,d)) for d in embed_dims])


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    # def to_3d(self,x):
    #     return rearrange(x, 'b h w c-> b (h w) c')

    def forward(self, x):
        x = self.patch_embed(x) #输入 x (B 64 64 96)

        outs = []

        for i in range(self.num_layers):
        # 添加频率自适应滤波器模块
            # n,H,W,Q_channel = x.shape
            # H,W = int(HW**0.5),int(HW**0.5)
            # query = x.view(n,H,W,Q_channel)
            query = x
            if i == 0: 
                query = self.DynamicFilter_1(query)
            elif i == 1:
                query = self.DynamicFilter_2(query)
            elif i == 2:
                query = self.DynamicFilter_3(query)
            elif i == 3:
                query = self.DynamicFilter_4(query)
            x = x + query
        #使用残差连接
            layer = self.layers[i]
            x_out, x = layer(x)
            
            # df = [self.DynamicFilter_1, self.DynamicFilter_2, self.DynamicFilter_3, self.DynamicFilter_4][i]
            # query = df(self.df_norms[i](x))
            # x = x + self.df_drop(self.df_gamma[i] * query)   # 不用 gamma 就去掉
            # x_out, x = self.layers[i](x)
            
            # 1: x_out (B 64 64 96) , x (B 32 32 192)   2: x_out (B 32 32 192) , x (B 16 16 384)    3: x_out (B 16 16 384) , x (B 8 8 768)  4: x_out (B 8 8 768) , x None
            if i in self.out_indices:
                x_out = self.extra_norms[i](x_out)
                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        
        return tuple(outs)

    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
