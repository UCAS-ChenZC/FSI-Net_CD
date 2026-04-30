# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Optional: MMCV CARAFE
# try:
#     from mmcv.ops import CARAFEPack
# except Exception:
#     CARAFEPack = None


# def _choose_heads(dim: int, max_heads: int = 4) -> int:
#     # pick a small head number that divides dim
#     for h in [max_heads, 3, 2, 1]:
#         if dim % h == 0:
#             return h
#     return 1


# class AlignUpsample(nn.Module):
#     """Content-aware upsample if CARAFE exists, else PixelShuffle."""
#     def __init__(self, channels: int, scale_factor: int = 2):
#         super().__init__()
#         self.scale = scale_factor
#         if CARAFEPack is not None:
#             # compressed_channels in CARAFE encoder should be <= channels typically
#             cc = min(64, channels)
#             self.up = CARAFEPack(
#                 channels=channels,
#                 scale_factor=scale_factor,
#                 up_kernel=5,
#                 up_group=1,
#                 encoder_kernel=3,
#                 encoder_dilation=1,
#                 compressed_channels=cc
#             )
#         else:
#             self.up = nn.Sequential(
#                 nn.Conv2d(channels, channels * (scale_factor ** 2), 3, padding=1, bias=False),
#                 nn.BatchNorm2d(channels * (scale_factor ** 2)),
#                 nn.ReLU(inplace=True),
#                 nn.PixelShuffle(scale_factor)
#             )

#     def forward(self, x):
#         return self.up(x)


# class LiftingStep(nn.Module):
#     """
#     One lifting step along a single dimension (width or height), depthwise linear predict/update.
#     Forward:
#         d = odd - P(even)
#         s = even + U(d)
#     Inverse:
#         even = s - U(d)
#         odd  = d + P(even)
#     """
#     def __init__(self, channels: int, k: int = 3, along: str = "width"):
#         super().__init__()
#         assert k % 2 == 1, "k must be odd"
#         assert along in ["width", "height"]
#         self.channels = channels
#         self.k = k
#         self.along = along

#         if along == "width":
#             ks = (1, k)
#             pad = (0, k // 2)
#         else:
#             ks = (k, 1)
#             pad = (k // 2, 0)

#         self.P = nn.Conv2d(channels, channels, kernel_size=ks, padding=pad,
#                            groups=channels, bias=False)
#         self.U = nn.Conv2d(channels, channels, kernel_size=ks, padding=pad,
#                            groups=channels, bias=False)

#         self._init_haar_like()

#     def _init_haar_like(self):
#         # Init to Haar lifting:
#         # Predict: P(even)=even  -> d = odd-even
#         # Update : U(d)=0.5*d   -> s = even + 0.5*d
#         with torch.no_grad():
#             self.P.weight.zero_()
#             self.U.weight.zero_()
#             c = self.channels
#             mid = self.k // 2
#             if self.along == "width":
#                 # weight shape: (C,1,1,k)
#                 self.P.weight[:, 0, 0, mid] = 1.0
#                 self.U.weight[:, 0, 0, mid] = 0.5
#             else:
#                 # weight shape: (C,1,k,1)
#                 self.P.weight[:, 0, mid, 0] = 1.0
#                 self.U.weight[:, 0, mid, 0] = 0.5

#     def forward(self, x):
#         # x: (B,C,H,W)
#         if self.along == "width":
#             assert x.size(-1) % 2 == 0, "Width must be even for lifting DWT."
#             even = x[..., 0::2]
#             odd  = x[..., 1::2]
#         else:
#             assert x.size(-2) % 2 == 0, "Height must be even for lifting DWT."
#             even = x[:, :, 0::2, :]
#             odd  = x[:, :, 1::2, :]

#         d = odd - self.P(even)
#         s = even + self.U(d)
#         return s, d

#     def inverse(self, s, d):
#         even = s - self.U(d)
#         odd  = d + self.P(even)

#         if self.along == "width":
#             B, C, H, W2 = even.shape
#             out = torch.empty((B, C, H, W2 * 2), device=even.device, dtype=even.dtype)
#             out[..., 0::2] = even
#             out[..., 1::2] = odd
#         else:
#             B, C, H2, W = even.shape
#             out = torch.empty((B, C, H2 * 2, W), device=even.device, dtype=even.dtype)
#             out[:, :, 0::2, :] = even
#             out[:, :, 1::2, :] = odd
#         return out


# class LearnableLiftingDWT2D(nn.Module):
#     """2D learnable (lifting-based) wavelet analysis: x -> (LL, LH, HL, HH)."""
#     def __init__(self, channels: int, k: int = 3):
#         super().__init__()
#         self.lift_w = LiftingStep(channels, k=k, along="width")
#         self.lift_h = LiftingStep(channels, k=k, along="height")

#     def forward(self, x):
#         # width lifting
#         s_w, d_w = self.lift_w(x)          # (B,C,H,W/2)
#         # height lifting on low-width band
#         LL, LH = self.lift_h(s_w)          # (B,C,H/2,W/2)
#         # height lifting on high-width band
#         HL, HH = self.lift_h(d_w)          # (B,C,H/2,W/2)
#         return LL, LH, HL, HH


# class LearnableLiftingIDWT2D(nn.Module):
#     """2D learnable (lifting-based) wavelet synthesis: (LL,LH,HL,HH) -> x."""
#     def __init__(self, channels: int, k: int = 3):
#         super().__init__()
#         self.lift_w = LiftingStep(channels, k=k, along="width")
#         self.lift_h = LiftingStep(channels, k=k, along="height")

#     def forward(self, LL, LH, HL, HH):
#         # inverse height first (reverse of forward)
#         s_w = self.lift_h.inverse(LL, LH)  # (B,C,H, W/2)
#         d_w = self.lift_h.inverse(HL, HH)  # (B,C,H, W/2)
#         # inverse width
#         x = self.lift_w.inverse(s_w, d_w)  # (B,C,2H,2W)
#         return x


# class SubbandRelationTransformer(nn.Module):
#     """Very lightweight transformer on 8 subband tokens (global pooled)."""
#     def __init__(self, dim: int, num_tokens: int = 8, heads: int = 4, mlp_ratio: float = 2.0):
#         super().__init__()
#         heads = _choose_heads(dim, heads)
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)

#         hidden = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, dim),
#         )
#         self.to_scale = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Sigmoid()
#         )
#         self.num_tokens = num_tokens

#     def forward(self, subbands):
#         # subbands: list of tensors length=8, each (B,C,H,W)
#         B, C = subbands[0].shape[:2]
#         tokens = [] #   每个token为(2 64) B C
#         for sb in subbands:
#             t = F.adaptive_avg_pool2d(sb, 1).view(B, C)  # (B,C)
#             tokens.append(t)
#         x = torch.stack(tokens, dim=1)  # (B,8,C)

#         y = self.norm1(x)   # LN
#         y, _ = self.attn(y, y, y, need_weights=False)
#         x = x + y

#         y = self.norm2(x)   # LN
#         y = self.mlp(y)
#         x = x + y

#         scales = self.to_scale(x)  # (B,8,C) in (0,1)
#         out = []
#         for i, sb in enumerate(subbands):
#             s = scales[:, i, :].view(B, C, 1, 1)
#             out.append(sb * s)
#         return out


# class SoftmaxFusionGate(nn.Module):
#     """
#     Competitive fusion for one subband:
#         cond = [hr, lr, |hr-lr|]
#         w = softmax(spatial_logits + channel_bias)
#         fused = w_hr*hr + w_lr*lr
#     """
#     def __init__(self, channels: int):
#         super().__init__()
#         in_ch = 3 * channels
#         self.spatial = nn.Sequential(
#             nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, 2, 1, bias=True)  # logits for (hr, lr)
#         )
#         self.channel = nn.Sequential(
#             nn.Linear(in_ch, max(in_ch // 4, 8)),
#             nn.ReLU(inplace=True),
#             nn.Linear(max(in_ch // 4, 8), 2)
#         )

#     def forward(self, hr_sb, lr_sb):
#         cond = torch.cat([hr_sb, lr_sb, (hr_sb - lr_sb).abs()], dim=1)
#         logits_spa = self.spatial(cond)  # (B,2,H,W)

#         gap = F.adaptive_avg_pool2d(cond, 1).flatten(1)  # (B,3C)
#         logits_chn = self.channel(gap).view(-1, 2, 1, 1)  # (B,2,1,1)

#         w = torch.softmax(logits_spa + logits_chn, dim=1)  # (B,2,H,W)
#         fused = w[:, 0:1] * hr_sb + w[:, 1:2] * lr_sb
#         return fused


# class LIFT_IFF(nn.Module):
#     """
#     LIFT-IFF: Lifting-based Interactive Frequency Fusion

#     Inputs:
#         hr_feat: (B, C_hr, 2H, 2W)
#         lr_feat: (B, C_lr, H, W)
#     Output:
#         fused:   (B, C_hr, 2H, 2W)
#     """
#     def __init__(self, hr_channels: int, lr_channels: int, reduction: int = 2,
#                  wavelet_k: int = 3):
#         super().__init__()
#         mid = max(hr_channels // reduction, 8)

#         # compression
#         self.hr_proj = nn.Sequential(
#             nn.Conv2d(hr_channels, mid, 1, bias=False),
#             nn.BatchNorm2d(mid),
#             nn.ReLU(inplace=True)
#         )
#         self.lr_proj = nn.Sequential(
#             nn.Conv2d(lr_channels, mid, 1, bias=False),
#             nn.BatchNorm2d(mid),
#             nn.ReLU(inplace=True)
#         )

#         # align LR -> HR resolution in mid-space
#         self.lr_up = AlignUpsample(mid, scale_factor=2)

#         # learnable invertible wavelet
#         self.dwt = LearnableLiftingDWT2D(mid, k=wavelet_k)
#         self.idwt = LearnableLiftingIDWT2D(mid, k=wavelet_k)

#         # subband relation modeling (8 tokens)
#         self.srt = SubbandRelationTransformer(dim=mid, heads=4, mlp_ratio=2.0)

#         # competitive fusion per subband
#         self.fuse_LL = SoftmaxFusionGate(mid)
#         self.fuse_LH = SoftmaxFusionGate(mid)
#         self.fuse_HL = SoftmaxFusionGate(mid)
#         self.fuse_HH = SoftmaxFusionGate(mid)

#         # optional high-band mixing (directional interaction)
#         self.high_mix = nn.Sequential(
#             nn.Conv2d(3 * mid, 3 * mid, 3, padding=1, groups=3 * mid, bias=False),
#             nn.BatchNorm2d(3 * mid),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(3 * mid, 3 * mid, 1, bias=False),
#             nn.BatchNorm2d(3 * mid),
#             nn.ReLU(inplace=True),
#         )

#         # output projection + LayerScale-like residual weights
#         self.out_proj = nn.Sequential(
#             nn.Conv2d(mid, hr_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(hr_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.lr_res_proj = nn.Sequential(
#             nn.Conv2d(mid, hr_channels, 1, bias=False),
#             nn.BatchNorm2d(hr_channels),
#         )

#         self.gamma_fused = nn.Parameter(torch.zeros(1))
#         self.gamma_lr = nn.Parameter(torch.zeros(1))

#     def forward(self, hr_feat, lr_feat):
#         B, C_hr, H2, W2 = hr_feat.shape
#         assert H2 % 2 == 0 and W2 % 2 == 0, "hr_feat spatial size must be even."
#         H, W = H2 // 2, W2 // 2
#         assert lr_feat.shape[-2:] == (H, W), "lr_feat must be half resolution of hr_feat."

#         hr = self.hr_proj(hr_feat)         # (B,mid,2H,2W) chanel:256 -> 64
#         lr = self.lr_proj(lr_feat)         # (B,mid,H,W)    chanel:256 -> 64
#         lr_up = self.lr_up(lr)             # (B,mid,2H,2W)  对lr进行CAFERE内容感知上采样 32 -> 64

#         # wavelet analysis at same (2H,2W) -> subbands at (H,W)
#         # 做小波变化分离出四个子带
#         LL_h, LH_h, HL_h, HH_h = self.dwt(hr)
#         LL_l, LH_l, HL_l, HH_l = self.dwt(lr_up)

#         # subband token relation transformer (global channel reweight)
#         subbands = [LL_h, LH_h, HL_h, HH_h, LL_l, LH_l, HL_l, HH_l]
#         LL_h, LH_h, HL_h, HH_h, LL_l, LH_l, HL_l, HH_l = self.srt(subbands)

#         # competitive fusion per band
#         LL = self.fuse_LL(LL_h, LL_l)
#         LH = self.fuse_LH(LH_h, LH_l)
#         HL = self.fuse_HL(HL_h, HL_l)
#         HH = self.fuse_HH(HH_h, HH_l)

#         # mix high bands jointly
#         high = torch.cat([LH, HL, HH], dim=1)
#         high = self.high_mix(high)
#         LH, HL, HH = torch.chunk(high, 3, dim=1)

#         # inverse wavelet back to (2H,2W)
#         fused_mid = self.idwt(LL, LH, HL, HH)  # (B,mid,2H,2W)

#         out = hr_feat + self.gamma_fused * self.out_proj(fused_mid) + self.gamma_lr * self.lr_res_proj(lr_up)
#         return out


# # ---------------- quick sanity test ----------------
# if __name__ == "__main__":
#     ###     VSCode tensor print setting     ###
#     def custom_repr(self):
#         return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

#     original_repr = torch.Tensor.__repr__
#     torch.Tensor.__repr__ = custom_repr
#     ###     VSCode tensor print setting     ###
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # print(f"Using device: {device}")
#     x_hr = torch.randn(2, 256, 64, 64).to(device)  # (B,C,2H,2W)
#     x_lr = torch.randn(2, 256, 32, 32).to(device)  # (B,C,H,W)
#     m = LIFT_IFF(hr_channels=256, lr_channels=256, reduction=4, wavelet_k=3).to(device)
#     y = m(x_hr, x_lr)
#     print(y.shape)  # expected: (2,256,64,64)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.carafe import carafe  # 确保你的 mmcv 编译/安装包含 carafe CUDA op
import numpy as np

# mmcv carafe (functional) is what FreqFusion uses
# try:
#     from mmcv.ops.carafe import carafe as mmcv_carafe  # mmcv <= 2.x usually
# except Exception:
#     mmcv_carafe = None

# import numpy as np


def _hamming2d(k: int) -> torch.Tensor:
    """(k,k) hamming window."""
    w1 = np.hamming(k).astype(np.float32)
    w2 = np.hamming(k).astype(np.float32)
    w = np.outer(w1, w2)  # (k,k)
    return torch.from_numpy(w)


def _init_xavier(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _init_small_std(m: nn.Module, std: float = 1e-3):
    # for mask generators (like FreqFusion)
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _kernel_normalizer(mask: torch.Tensor, k: int, hamming_flat: torch.Tensor) -> torch.Tensor:
    """
    mask: (B, k*k * M, H, W) where M includes group * scale^2
    normalize over k*k for each channel-group position, apply hamming window.
    """
    B, C, H, W = mask.shape
    assert C % (k * k) == 0
    M = C // (k * k)

    mask = mask.view(B, M, k * k, H, W)
    mask = torch.softmax(mask, dim=2)

    # hamming_flat: (k*k,)
    mask = mask * hamming_flat.view(1, 1, k * k, 1, 1)
    mask = mask / (mask.sum(dim=2, keepdim=True) + 1e-6)

    mask = mask.view(B, M * k * k, H, W).contiguous()
    return mask


# def _safe_carafe(x: torch.Tensor, mask: torch.Tensor, k: int, group: int, scale: int) -> torch.Tensor:
#     """
#     CARAFE is usually CUDA-only. If unavailable / CPU -> fallback to interpolate (+ optional smooth).
#     """
#     if (mmcv_carafe is not None) and x.is_cuda:
#         return mmcv_carafe(x, mask, k, group, scale)
#     # CPU / no-op fallback
#     if scale != 1:
#         x = F.interpolate(x, scale_factor=scale, mode='nearest')
#     return x


class LiftingStep(nn.Module):
    """Same lifting step as before, kept minimal."""
    def __init__(self, channels: int, k: int = 3, along: str = "width"):
        super().__init__()
        assert k % 2 == 1
        assert along in ["width", "height"]
        if along == "width":
            ks, pad = (1, k), (0, k // 2)
        else:
            ks, pad = (k, 1), (k // 2, 0)

        self.along = along
        self.P = nn.Conv2d(channels, channels, ks, padding=pad, groups=channels, bias=False)
        self.U = nn.Conv2d(channels, channels, ks, padding=pad, groups=channels, bias=False)

        # haar-like init
        with torch.no_grad():
            self.P.weight.zero_()
            self.U.weight.zero_()
            mid = k // 2
            if along == "width":
                self.P.weight[:, 0, 0, mid] = 1.0
                self.U.weight[:, 0, 0, mid] = 0.5
            else:
                self.P.weight[:, 0, mid, 0] = 1.0
                self.U.weight[:, 0, mid, 0] = 0.5

    def forward(self, x):
        if self.along == "width":
            assert x.size(-1) % 2 == 0
            even, odd = x[..., 0::2], x[..., 1::2]
        else:
            assert x.size(-2) % 2 == 0
            even, odd = x[:, :, 0::2, :], x[:, :, 1::2, :]

        d = odd - self.P(even)
        s = even + self.U(d)
        return s, d

    def inverse(self, s, d):
        even = s - self.U(d)
        odd = d + self.P(even)

        if self.along == "width":
            B, C, H, W2 = even.shape
            out = torch.empty((B, C, H, W2 * 2), device=even.device, dtype=even.dtype)
            out[..., 0::2] = even
            out[..., 1::2] = odd
        else:
            B, C, H2, W = even.shape
            out = torch.empty((B, C, H2 * 2, W), device=even.device, dtype=even.dtype)
            out[:, :, 0::2, :] = even
            out[:, :, 1::2, :] = odd
        return out


class LearnableLiftingDWT2D(nn.Module):
    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.lw = LiftingStep(channels, k=k, along="width")
        self.lh = LiftingStep(channels, k=k, along="height")

    def forward(self, x):
        s_w, d_w = self.lw(x)
        LL, LH = self.lh(s_w)
        HL, HH = self.lh(d_w)
        return LL, LH, HL, HH


class LearnableLiftingIDWT2D(nn.Module):
    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.lw = LiftingStep(channels, k=k, along="width")
        self.lh = LiftingStep(channels, k=k, along="height")

    def forward(self, LL, LH, HL, HH):
        s_w = self.lh.inverse(LL, LH)
        d_w = self.lh.inverse(HL, HH)
        x = self.lw.inverse(s_w, d_w)
        return x


class SoftmaxFusionGate(nn.Module):
    """same idea: [hr, lr, |hr-lr|] -> softmax weights -> fuse"""
    def __init__(self, channels: int):
        super().__init__()
        in_ch = 3 * channels
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1, bias=True),
        )
        self.channel = nn.Sequential(
            nn.Linear(in_ch, max(in_ch // 4, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_ch // 4, 8), 2),
        )

    def forward(self, hr_sb, lr_sb):
        cond = torch.cat([hr_sb, lr_sb, (hr_sb - lr_sb).abs()], dim=1)
        logits_spa = self.spatial(cond)
        gap = F.adaptive_avg_pool2d(cond, 1).flatten(1)
        logits_chn = self.channel(gap).view(-1, 2, 1, 1)
        w = torch.softmax(logits_spa + logits_chn, dim=1)
        return w[:, 0:1] * hr_sb + w[:, 1:2] * lr_sb


class FreqWaveletFusion(nn.Module):
    """
    FreqFusion-style ALPF/AHPF (CARAFE) + Learnable lifting-wavelet subband fusion.
    Return:
        mask_lr, hr_out, lr_up_out
    """
    def __init__(self,
                 hr_channels: int,
                 lr_channels: int,
                 compressed_channels: int = 64,
                 lowpass_kernel: int = 5,
                 highpass_kernel: int = 3,
                 up_group: int = 1,
                 wavelet_k: int = 3,
                 reduction: int = 4,
                 hr_residual: bool = True,
                 hamming_window: bool = True,
                 scale: int = 2):
        super().__init__()
        assert scale == 2, "当前实现按 hr=2x lr 写死为2（与你的金字塔用法一致）"
        self.scale = scale

        self.hr_channels = hr_channels
        self.lr_channels = lr_channels
        self.cc = compressed_channels
        self.lp_k = lowpass_kernel
        self.hp_k = highpass_kernel
        self.up_group = up_group
        self.hr_residual = hr_residual

        # --- compressors (same spirit as FreqFusion) ---
        self.hr_comp = nn.Conv2d(hr_channels, compressed_channels, 1)
        self.lr_comp = nn.Conv2d(lr_channels, compressed_channels, 1)

        # --- mask generators ---
        # NOTE: for carafe(scale=2), mask channel must be k^2 * group * 4
        # self.alpf = nn.Conv2d(compressed_channels, (lowpass_kernel ** 2) * up_group * (scale ** 2),
        #                       kernel_size=3, padding=1)
        self.alpf = nn.Conv2d(compressed_channels, (lowpass_kernel ** 2) * up_group, 3, padding=1)
        self.ahpf = nn.Conv2d(compressed_channels, (highpass_kernel ** 2) * up_group,
                              kernel_size=3, padding=1)

        # --- hamming window buffers (match FreqFusion style) ---
        if hamming_window:
            lp_ham = _hamming2d(lowpass_kernel).float()[None, None, ...]   # (1,1,k,k)
            hp_ham = _hamming2d(highpass_kernel).float()[None, None, ...]
        else:
            lp_ham = torch.ones(1, 1, lowpass_kernel, lowpass_kernel)
            hp_ham = torch.ones(1, 1, highpass_kernel, highpass_kernel)

        self.register_buffer("hamming_lowpass", lp_ham, persistent=False)
        self.register_buffer("hamming_highpass", hp_ham, persistent=False)

        # --- wavelet fusion path (mid channels) ---
        mid = max(hr_channels // reduction, 16)
        self.hr_mid = nn.Sequential(
            nn.Conv2d(hr_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.lr_mid = nn.Sequential(
            nn.Conv2d(lr_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        self.dwt = LearnableLiftingDWT2D(mid, k=wavelet_k)
        self.idwt = LearnableLiftingIDWT2D(mid, k=wavelet_k)

        self.fuse_LL = SoftmaxFusionGate(mid)
        self.fuse_LH = SoftmaxFusionGate(mid)
        self.fuse_HL = SoftmaxFusionGate(mid)
        self.fuse_HH = SoftmaxFusionGate(mid)

        self.high_mix = nn.Sequential(
            nn.Conv2d(3 * mid, 3 * mid, 3, padding=1, groups=3 * mid, bias=False),
            nn.BatchNorm2d(3 * mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(3 * mid, 3 * mid, 1, bias=False),
            nn.BatchNorm2d(3 * mid),
            nn.ReLU(inplace=True),
        )

        self.to_hr = nn.Sequential(
            nn.Conv2d(mid, hr_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hr_channels),
            nn.ReLU(inplace=True),
        )
        self.to_lr = nn.Sequential(
            nn.Conv2d(mid, lr_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(lr_channels),
            nn.ReLU(inplace=True),
        )

        self.gamma_hr = nn.Parameter(torch.zeros(1))
        self.gamma_lr = nn.Parameter(torch.zeros(1))

        # init similar to your style
        self.apply(_init_xavier)
        _init_small_std(self.alpf, std=1e-3)
        _init_small_std(self.ahpf, std=1e-3)

    @staticmethod
    def kernel_normalizer(mask: torch.Tensor, kernel: int, hamming: torch.Tensor) -> torch.Tensor:
        """
        Follow the same reshaping idea as your attached FreqFusion.kernel_normalizer.
        mask: (N, kernel^2 * M, H, W)
        """
        n, mask_c, h, w = mask.size()
        assert mask_c % (kernel * kernel) == 0
        mask_channel = int(mask_c / float(kernel ** 2))

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)

        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).reshape(n, -1, kernel, kernel)

        mask = mask * hamming.to(mask.device, dtype=mask.dtype)  # (1,1,k,k) broadcast
        mask = mask / (mask.sum(dim=(-1, -2), keepdim=True) + 1e-6)

        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).reshape(n, -1, h, w).contiguous()
        return mask

    def forward(self, hr_feat: torch.Tensor, lr_feat: torch.Tensor):
        """
        hr_feat: (B, C_hr, 2H, 2W)
        lr_feat: (B, C_lr, H, W)
        """
        B, _, H2, W2 = hr_feat.shape
        H, W = lr_feat.shape[-2:]
        assert (H2, W2) == (2 * H, 2 * W), "Expect hr spatial = 2x lr spatial."
        assert lr_feat.size(1) % self.up_group == 0, "lr_channels must be divisible by up_group"
        assert hr_feat.size(1) % self.up_group == 0, "hr_channels must be divisible by up_group"

        # -------------------------
        # (A) build masks on HR resolution (2H,2W)  —— 这是你报错的关键修正点
        # -------------------------
        chr = self.hr_comp(hr_feat)      # (B,cc,2H,2W)     cc: compressed channels 256->64
        clr = self.lr_comp(lr_feat)      # (B,cc,H,W)

        # 1) AHPF init (on compressed_hr_feat), then update compressed_hr_feat (FreqFusion style)
        mask_hr_hr = self.ahpf(chr)  # (B, hp_k^2*group, 2H,2W)
        mask_hr_init = self.kernel_normalizer(mask_hr_hr, self.hp_k, self.hamming_highpass)
        # compressed_hr_feat = x + x - LP(x)  (same structure as attachment)
        chr = chr + chr - carafe(chr, mask_hr_init, self.hp_k, self.up_group, 1)

        # 2) low-pass mask from updated chr (HR scale)
        mask_lr_hr = self.alpf(chr)  # (B, lp_k^2*group*4, 2H,2W)
        mask_lr_init = self.kernel_normalizer(mask_lr_hr, self.lp_k, self.hamming_lowpass)

        # 3) also compute low-pass mask features from LR side, then upsample them guided by HR mask
        mask_lr_lr_lr = self.alpf(clr)  # (B, lp_k^2*group*4, H,W)  # ✅ (B, lp_k^2*group, H, W)
        mask_lr_lr = carafe(mask_lr_lr_lr, mask_lr_init, self.lp_k, self.up_group, 2)  # -> (2H,2W)

        mask_lr = mask_lr_hr + mask_lr_lr
        mask_lr = self.kernel_normalizer(mask_lr, self.lp_k, self.hamming_lowpass)

        # upsample lr_feat by 2 using the HR-sized mask
        
        lr_up = carafe(lr_feat, mask_lr, self.lp_k, self.up_group, 2)  # (B,C_lr,2H,2W)

        # 4) high-pass mask fusion (hr-side + lr-side upsampled), then apply to hr_feat
        mask_hr_lr_lr = self.ahpf(clr)  # (B, hp_k^2*group, H,W)
        # use lowpass mask to upsample this "highpass-mask-feature" like the attachment does
        mask_hr_lr = carafe(mask_hr_lr_lr, mask_lr, self.lp_k, self.up_group, 2)  # (2H,2W)

        mask_hr = mask_hr_hr + mask_hr_lr
        mask_hr = self.kernel_normalizer(mask_hr, self.hp_k, self.hamming_highpass)

        hr_lp = carafe(hr_feat, mask_hr, self.hp_k, self.up_group, 1)
        hr_hf = hr_feat - hr_lp
        hr_enh = hr_feat + hr_hf if self.hr_residual else hr_hf

        # -------------------------
        # (B) lifting-wavelet subband competitive fusion on aligned hr_enh & lr_up
        # -------------------------
        hr_m = self.hr_mid(hr_enh)   # (B,mid,2H,2W)
        lr_m = self.lr_mid(lr_up)    # (B,mid,2H,2W)

        LL_h, LH_h, HL_h, HH_h = self.dwt(hr_m)   # (B,mid,H,W)
        LL_l, LH_l, HL_l, HH_l = self.dwt(lr_m)

        LL = self.fuse_LL(LL_h, LL_l)
        LH = self.fuse_LH(LH_h, LH_l)
        HL = self.fuse_HL(HL_h, HL_l)
        HH = self.fuse_HH(HH_h, HH_l)

        high = torch.cat([LH, HL, HH], dim=1)
        high = self.high_mix(high)
        LH, HL, HH = torch.chunk(high, 3, dim=1)

        fused_mid = self.idwt(LL, LH, HL, HH)  # (B,mid,2H,2W)

        hr_out = hr_enh + self.gamma_hr * self.to_hr(fused_mid)
        lr_up_out = lr_up + self.gamma_lr * self.to_lr(fused_mid)

        return mask_lr, hr_out, lr_up_out


# ---------------- quick sanity test ----------------
if __name__ == "__main__":
    ###     VSCode tensor print setting     ###
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr
    ###     VSCode tensor print setting     ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_hr = torch.randn(1, 256, 16, 16).to(device)
    x_lr = torch.randn(1, 256, 8, 8).to(device)
    m = FreqWaveletFusion(256, 256, compressed_channels=64, reduction=4).to(device)
    print("cuda_available:", torch.cuda.is_available())
    print("x_hr.is_cuda:", x_hr.is_cuda, "x_lr.is_cuda:", x_lr.is_cuda)
    print("model.is_cuda:", next(m.parameters()).is_cuda)

    mask, hro, lru = m(x_hr, x_lr)
    print(mask.shape, hro.shape, lru.shape)  # (B, lp_k^2*group*4, H, W), (B,256,64,64), (B,512,64,64)
