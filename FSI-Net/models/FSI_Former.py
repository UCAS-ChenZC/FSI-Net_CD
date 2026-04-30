
import math
import os
import sys
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# -----------------------------------------------------------------------------
# Optional timm dependency (DropPath + trunc_normal_)
# -----------------------------------------------------------------------------
try:
    from timm.models.layers import DropPath, trunc_normal_  # type: ignore
except Exception:
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or (not self.training):
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        # Fallback to torch.nn.init.trunc_normal_ if available
        if hasattr(torch.nn.init, "trunc_normal_"):
            return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
        # Very simple fallback: normal then clamp (not exact truncated normal)
        with torch.no_grad():
            tensor.normal_(mean, std)
            tensor.clamp_(min=a, max=b)
        return tensor

# -----------------------------------------------------------------------------
# Optional mmseg/mmcv dependency (keep your original imports if available)
# -----------------------------------------------------------------------------
try:
    from ..builder import BACKBONES  # type: ignore
except Exception:
    BACKBONES = None  # allow standalone import

try:
    from mmcv_custom import load_checkpoint  # type: ignore
    from mmseg.utils import get_root_logger  # type: ignore
except Exception:
    load_checkpoint = None
    get_root_logger = None
    
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

        # --------  --------
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
        """
        x: (b h w c)
        """
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)           # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x


# =============================================================================
# Frequency-domain components (NEW)
# =============================================================================
class FourierGlobalFilter2d(nn.Module):
    """
    GFNet-style global filter:
      FFT2 -> element-wise multiply with learnable complex filter -> iFFT2

    - Resolution-adaptive: complex filter weights are defined on a base resolution,
      then bilinearly interpolated to current (H, W//2+1) frequency map.

    Args:
        dim: channel dimension (C)
        base_resolution: spatial resolution (Hb, Wb) used to parametrize the filter.
        energy_thresh: radial frequency threshold to compute (low/high) energy stats.
    """
    def __init__(
        self,
        dim: int,
        base_resolution: Tuple[int, int] = (56, 56),
        energy_thresh: float = 0.25,
    ):
        super().__init__()
        hb, wb = base_resolution
        self.base_resolution = (int(hb), int(wb))
        self.energy_thresh = float(energy_thresh)

        # Parametrized on rFFT output size: (Hb, Wb//2+1, C, 2)
        self.complex_weight = nn.Parameter(
            torch.randn(hb, wb // 2 + 1, dim, 2) * 0.02
        )

        # cache for radial grids (per device)
        self._radial_cache = {}  # key: (H, W, device) -> radial grid (H, W//2+1)

    def _resize_complex_weight(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """
        Return complex weight with shape (H, W//2+1, C) as complex tensor.
        """
        hb, wb = self.base_resolution
        Wf = W // 2 + 1

        w = self.complex_weight  # (hb, wb//2+1, C, 2) float
        # reshape to (1, 2C, hb, wb//2+1) for interpolation
        w = w.permute(2, 3, 0, 1).contiguous()  # (C, 2, hb, wb//2+1)
        w = w.view(1, -1, hb, wb // 2 + 1)      # (1, 2C, hb, wb//2+1)
        if (hb != H) or (wb // 2 + 1 != Wf):
            w = F.interpolate(w, size=(H, Wf), mode="bilinear", align_corners=False)

        # back to (H, Wf, C, 2)
        C2 = w.shape[1]
        dim = C2 // 2
        w = w.view(dim, 2, H, Wf).permute(2, 3, 0, 1).contiguous()
        w = torch.view_as_complex(w)  # (H, Wf, C) complex
        return w.to(device=device)

    def _get_radial_grid(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """
        Radial frequency grid aligned with torch.fft.rfft2 output:
          height uses fftfreq(H), width uses rfftfreq(W).
        Returns: (H, W//2+1) in [0, ~0.707]
        """
        key = (H, W, str(device))
        if key in self._radial_cache:
            return self._radial_cache[key].to(device=device, dtype=dtype)

        fy = torch.fft.fftfreq(H, d=1.0, device=device)[:, None]      # (H, 1), includes negative
        fx = torch.fft.rfftfreq(W, d=1.0, device=device)[None, :]     # (1, W//2+1), non-negative
        r = torch.sqrt(fy ** 2 + fx ** 2)                             # (H, W//2+1)
        # cache on CPU to avoid GPU memory blow-up; move to device in return
        self._radial_cache[key] = r.detach().cpu()
        return r.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        """
        x: (B, H, W, C)
        return:
            y: (B, H, W, C)
            stats (optional): (B, 2) = [hf_ratio, log_total_energy]
        """
        B, H, W, C = x.shape
        orig_dtype = x.dtype

        # FFT is safer in fp32 (esp. with AMP), so cast
        x_fp32 = x.float()
        x_fft = torch.fft.rfft2(x_fp32, s=(H, W), dim=(1, 2), norm="ortho")  # (B, H, W//2+1, C), complex

        weight = self._resize_complex_weight(H, W, device=x_fft.device, dtype=x_fp32.dtype)  # (H, Wf, C), complex
        y_fft = x_fft * weight[None, :, :, :]  # broadcast over batch
        y = torch.fft.irfft2(y_fft, s=(H, W), dim=(1, 2), norm="ortho")
        y = y.to(dtype=orig_dtype)

        if not return_stats:
            return y

        # energy stats from ORIGINAL spectrum (before filtering)
        r = self._get_radial_grid(H, W, device=x_fft.device, dtype=x_fp32.dtype)
        low_mask = (r <= self.energy_thresh).to(x_fp32.dtype)   # (H, Wf)
        high_mask = 1.0 - low_mask

        mag2 = (x_fft.real ** 2 + x_fft.imag ** 2)  # (B, H, Wf, C)
        low_e = (mag2 * low_mask[None, :, :, None]).mean(dim=(1, 2, 3))
        high_e = (mag2 * high_mask[None, :, :, None]).mean(dim=(1, 2, 3))
        total = low_e + high_e + 1e-6
        hf_ratio = high_e / total
        log_total = torch.log(total)

        stats = torch.stack([hf_ratio, log_total], dim=-1).to(dtype=orig_dtype)
        return y, stats


class ChannelGate(nn.Module):
    """Squeeze-and-Excitation style channel gate for (B,H,W,C) tensors."""
    def __init__(self, dim: int, hidden_ratio: float = 0.25, eps: float = 1e-6):
        super().__init__()
        hidden = max(8, int(dim * hidden_ratio))
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        g = x.mean(dim=(1, 2))        # (B, C)
        g = self.fc(self.norm(g))     # (B, C)
        return g[:, None, None, :]    # (B, 1, 1, C)


# =============================================================================
# Retention positional encoding (same as your original)
# =============================================================================
class RetNetRelPos2d(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
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
        grid = torch.meshgrid([index_h, index_w], indexing='ij')
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # (H*W,2)
        mask = grid[:, None, :] - grid[None, :, :]          # (H*W,H*W,2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]             # (n, H*W, H*W)
        return mask

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l,l)
        mask = mask.abs()
        mask = mask * self.decay[:, None, None] # (n,l,l)
        return mask

    def forward(self, slen: Tuple[int, int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)

            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])
            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            mask = self.generate_2d_decay(slen[0], slen[1])
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


# =============================================================================
# Vision Retention (modified: support head_scale)
# =============================================================================
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
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos,
        chunkwise_recurrent: bool = False,
        incremental_state=None,
        head_scale: Optional[torch.Tensor] = None,
    ):
        """
        x: (b h w c)
        rel_pos: ((sin, cos), (mask_h, mask_w))
        head_scale: optional tensor:
            - (num_heads,) or (B, num_heads), used to scale the decay masks per head.
        """
        bsz, h, w, _ = x.size()
        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        qr_w = qr.transpose(1, 2)  # (b h n w d1)
        kr_w = kr.transpose(1, 2)  # (b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # (b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # (b h n w w)

        if head_scale is None:
            qk_mat_w = qk_mat_w + mask_w
        else:
            if head_scale.dim() == 1:
                scaled = mask_w * head_scale[:, None, None]  # (n, w, w)
                qk_mat_w = qk_mat_w + scaled
            else:
                # (B, n)
                scaled = mask_w[None, :, :, :] * head_scale[:, :, None, None]  # (B, n, w, w)
                qk_mat_w = qk_mat_w + scaled[:, None, :, :, :]                 # (B,1,n,w,w) -> broadcast over h

        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)  # (b h n w d2)

        qr_h = qr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        v = v.permute(0, 3, 2, 1, 4)      # (b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # (b w n h h)

        if head_scale is None:
            qk_mat_h = qk_mat_h + mask_h
        else:
            if head_scale.dim() == 1:
                scaled = mask_h * head_scale[:, None, None]  # (n, h, h)
                qk_mat_h = qk_mat_h + scaled
            else:
                scaled = mask_h[None, :, :, :] * head_scale[:, :, None, None]  # (B, n, h, h)
                qk_mat_h = qk_mat_h + scaled[:, None, :, :, :]                 # (B,1,n,h,h) -> broadcast over w

        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)  # (b w n h d2)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (b h w n*d2)
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
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos,
        chunkwise_recurrent: bool = False,
        incremental_state=None,
        head_scale: Optional[torch.Tensor] = None,
    ):
        """
        x: (b h w c)
        rel_pos: ((sin, cos), mask) where mask: (n, l, l)
        head_scale: optional tensor:
            - (num_heads,) or (B, num_heads), used to scale the decay mask per head.
        """
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        assert h * w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        qr = qr.flatten(2, 3)  # (b n l d1)
        kr = kr.flatten(2, 3)  # (b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4).flatten(2, 3)  # (b n l d2)

        qk_mat = qr @ kr.transpose(-1, -2)  # (b n l l)

        if head_scale is None:
            qk_mat = qk_mat + mask
        else:
            if head_scale.dim() == 1:
                qk_mat = qk_mat + mask * head_scale[:, None, None]
            else:
                qk_mat = qk_mat + mask[None, :, :, :] * head_scale[:, :, None, None]

        qk_mat = torch.softmax(qk_mat, -1)
        output = torch.matmul(qk_mat, vr)  # (b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)  # (b h w n*d2)
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
        subconv=True,
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

    def forward(self, x: torch.Tensor):
        """
        x: (b h w c)
        """
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
    def __init__(
        self,
        retention: str,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        drop_path=0.0,
        layerscale=False,
        layer_init_values=1e-5,
    ):
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
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(self, x: torch.Tensor, incremental_state=None, chunkwise_recurrent=False, retention_rel_pos=None):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x


# =============================================================================
# NEW: Frequency-Enhanced Retention Block
# =============================================================================
class RetBlockFreq(nn.Module):
    """
    A dual-domain block:
      - Spatial domain: Manhattan retention (your original)
      - Frequency domain: GFNet-style global Fourier filter (FFT->filter->iFFT)
      - Fusion: gated freq residual + frequency-driven head-scale on decay masks

    This is designed to be a drop-in replacement of RetBlock.
    """
    def __init__(
        self,
        retention: str,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        drop_path: float = 0.0,
        layerscale: bool = False,
        layer_init_values: float = 1e-5,
        # frequency cfg
        freq_base_resolution: Tuple[int, int] = (56, 56),
        freq_energy_thresh: float = 0.25,
        freq_residual_scale: float = 0.5,
        head_scale_range: Tuple[float, float] = (0.5, 1.5),
        gate_hidden_ratio: float = 0.25,
    ):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # ---- retention ----
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = VisionRetentionChunk(embed_dim, num_heads)
        else:
            self.retention = VisionRetentionAll(embed_dim, num_heads)

        # ---- frequency branch ----
        self.freq_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.freq_filter = FourierGlobalFilter2d(
            dim=embed_dim,
            base_resolution=freq_base_resolution,
            energy_thresh=freq_energy_thresh,
        )
        self.freq_gate = ChannelGate(embed_dim, hidden_ratio=gate_hidden_ratio, eps=1e-6)
        self.freq_scale = nn.Parameter(torch.tensor(float(freq_residual_scale)))

        # ---- frequency-driven head scale (range adaptation) ----
        hs_hidden = 32
        self.head_scale_mlp = nn.Sequential(
            nn.Linear(2, hs_hidden),
            nn.GELU(),
            nn.Linear(hs_hidden, num_heads),
            nn.Sigmoid()
        )
        self.head_scale_range = (float(head_scale_range[0]), float(head_scale_range[1]))

        # ---- FFN etc ----
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def _compute_head_scale(self, stats: torch.Tensor) -> torch.Tensor:
        """
        stats: (B, 2) = [hf_ratio, log_total_energy]
        return: head_scale: (B, num_heads) in [min,max]
        """
        hs = self.head_scale_mlp(stats.float())  # (B, n), in [0,1]
        min_s, max_s = self.head_scale_range
        hs = min_s + (max_s - min_s) * hs
        return hs

    def forward(self, x: torch.Tensor, incremental_state=None, chunkwise_recurrent=False, retention_rel_pos=None):
        """
        x: (B,H,W,C)
        """
        x = x + self.pos(x)

        # frequency branch (one FFT per block)
        freq_out, stats = self.freq_filter(self.freq_norm(x), return_stats=True)  # (B,H,W,C), (B,2)
        freq_out = self.freq_gate(freq_out) * freq_out  # channel-gated

        head_scale = self._compute_head_scale(stats)     # (B, num_heads)

        # retention with frequency-driven head scaling
        ret_out = self.retention(
            self.retention_layer_norm(x),
            retention_rel_pos,
            chunkwise_recurrent,
            incremental_state,
            head_scale=head_scale,
        )

        fused = ret_out + self.freq_scale * freq_out

        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * fused)
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(fused)
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.SyncBatchNorm(out_dim)

    def forward(self, x):
        """
        x: B H W C
        """
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.reduction(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        return x


class BasicLayer(nn.Module):
    """
    NOTE: Extended with frequency options (use_frequency, freq_*)
    """
    def __init__(
        self,
        embed_dim,
        out_dim,
        depth,
        num_heads,
        init_value: float,
        heads_range: float,
        ffn_dim=96.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        chunkwise_recurrent=False,
        downsample: PatchMerging = None,
        use_checkpoint=False,
        layerscale=False,
        layer_init_values=1e-5,
        # NEW
        use_frequency: bool = False,
        freq_base_resolution: Tuple[int, int] = (56, 56),
        freq_energy_thresh: float = 0.25,
        freq_residual_scale: float = 0.5,
        head_scale_range: Tuple[float, float] = (0.5, 1.5),
    ):
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
        Block = RetBlockFreq if use_frequency else RetBlock
        self.blocks = nn.ModuleList([
            Block(
                flag,
                embed_dim,
                num_heads,
                ffn_dim,
                drop_path[i] if isinstance(drop_path, list) else drop_path,
                layerscale,
                layer_init_values,
                # freq cfg (ignored by RetBlock)
                freq_base_resolution=freq_base_resolution,
                freq_energy_thresh=freq_energy_thresh,
                freq_residual_scale=freq_residual_scale,
                head_scale_range=head_scale_range,
            )
            if use_frequency else
            Block(
                flag,
                embed_dim,
                num_heads,
                ffn_dim,
                drop_path[i] if isinstance(drop_path, list) else drop_path,
                layerscale,
                layer_init_values,
            )
            for i in range(depth)
        ])

        # patch merging layer
        self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer) if downsample is not None else None
        self.SFIB_1 = SFIBv2(96)
        self.SFIB_2 = SFIBv2(192)
        self.SFIB_3 = SFIBv2(384)
        self.SFIB_4 = SFIBv2(768)

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
        
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(
                    blk,
                    x=x,
                    incremental_state=None,
                    chunkwise_recurrent=self.chunkwise_recurrent,
                    retention_rel_pos=rel_pos,
                )
            else:
                x = blk(x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)

        # x = x + query
        # identity = x.permute(0, 3, 1, 2)
        # if d == 96:
        #     query = self.SFIB_1(identity)
        # elif d == 192:
        #     query = self.SFIB_2(identity)
        # elif d == 384:
        #     query = self.SFIB_3(identity)
        # elif d == 768:
        #     query = self.SFIB_4(identity)
        # query = query.permute(0, 2, 3, 1)
        # x = query

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
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim),
        )

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        return x


# -----------------------------------------------------------------------------
# Original RMT backbone (kept)
# -----------------------------------------------------------------------------
# if BACKBONES is not None:
#     @BACKBONES.register_module()
#     class RMT(nn.Module):
#         def __init__(
#             self,
#             in_chans=3,
#             out_indices=(0, 1, 2, 3),
#             embed_dims=[96, 192, 384, 768],
#             depths=[2, 2, 6, 2],
#             num_heads=[3, 6, 12, 24],
#             init_values=[1, 1, 1, 1],
#             heads_ranges=[3, 3, 3, 3],
#             mlp_ratios=[3, 3, 3, 3],
#             drop_path_rate=0.1,
#             norm_layer=nn.LayerNorm,
#             patch_norm=True,
#             use_checkpoint=False,
#             chunkwise_recurrents=[True, True, False, False],
#             projection=1024,
#             layerscales=[False, False, False, False],
#             layer_init_values=1e-6,
#             norm_eval=True,
#         ):
#             super().__init__()
#             self.out_indices = out_indices
#             self.num_layers = len(depths)
#             self.embed_dim = embed_dims[0]
#             self.patch_norm = patch_norm
#             self.num_features = embed_dims[-1]
#             self.mlp_ratios = mlp_ratios
#             self.norm_eval = norm_eval

#             self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=norm_layer if self.patch_norm else None)

#             dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

#             self.layers = nn.ModuleList()
#             for i_layer in range(self.num_layers):
#                 layer = BasicLayer(
#                     embed_dim=embed_dims[i_layer],
#                     out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
#                     depth=depths[i_layer],
#                     num_heads=num_heads[i_layer],
#                     init_value=init_values[i_layer],
#                     heads_range=heads_ranges[i_layer],
#                     ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
#                     drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
#                     norm_layer=norm_layer,
#                     chunkwise_recurrent=chunkwise_recurrents[i_layer],
#                     downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                     use_checkpoint=use_checkpoint,
#                     layerscale=layerscales[i_layer],
#                     layer_init_values=layer_init_values,
#                     # frequency disabled by default
#                     use_frequency=False,
#                 )
#                 self.layers.append(layer)

#             self.extra_norms = nn.ModuleList([nn.LayerNorm(embed_dims[i]) for i in range(4)])
#             self.apply(self._init_weights)

#         def _init_weights(self, m):
#             if isinstance(m, nn.Linear):
#                 trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 try:
#                     nn.init.constant_(m.bias, 0)
#                     nn.init.constant_(m.weight, 1.0)
#                 except Exception:
#                     pass

#         def init_weights(self, pretrained=None):
#             def _init_weights(m):
#                 if isinstance(m, nn.Linear):
#                     trunc_normal_(m.weight, std=0.02)
#                     if m.bias is not None:
#                         nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.LayerNorm):
#                     nn.init.constant_(m.bias, 0)
#                     nn.init.constant_(m.weight, 1.0)

#             if isinstance(pretrained, str):
#                 self.apply(_init_weights)
#                 if (load_checkpoint is None) or (get_root_logger is None):
#                     raise RuntimeError("mmcv/mmseg is required for loading checkpoints in this file.")
#                 logger = get_root_logger()
#                 load_checkpoint(self, pretrained, strict=False, logger=logger)
#             elif pretrained is None:
#                 self.apply(_init_weights)
#             else:
#                 raise TypeError("pretrained must be a str or None")

#         def forward(self, x):
#             x = self.patch_embed(x)
#             outs = []
#             for i in range(self.num_layers):
#                 layer = self.layers[i]
#                 x_out, x = layer(x)
#                 if i in self.out_indices:
#                     x_out = self.extra_norms[i](x_out)
#                     out = x_out.permute(0, 3, 1, 2).contiguous()
#                     outs.append(out)
#             return tuple(outs)

#         def train(self, mode=True):
#             super().train(mode)
#             if mode and self.norm_eval:
#                 for m in self.modules():
#                     if isinstance(m, nn.BatchNorm2d):
#                         m.eval()


# -----------------------------------------------------------------------------
# NEW backbone: RMT_Freq (frequency-enhanced)
# -----------------------------------------------------------------------------
# if BACKBONES is not None:
#     @BACKBONES.register_module()
class RMT_Freq(nn.Module):
    """
    Drop-in RMT variant with frequency-enhanced blocks.

    Key extra args:
        - use_frequencies: per-stage bool list
        - freq_base_resolutions: per-stage base spatial resolution for spectral filters
        - freq_energy_thresh: low/high split threshold (normalized frequency)
        - freq_residual_scale: initial scale for freq residual (learnable)
        - head_scale_range: range to scale decay masks per head (min,max)
    """
    def __init__(
        self,
        in_chans=32,
        out_indices=(0, 1, 2, 3),
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
        layer_init_values=1e-6,
        norm_eval=True,
        # NEW
        use_frequencies: List[bool] = [True, True, False, False],
        freq_base_resolutions: Optional[List[Tuple[int, int]]] = None,
        freq_energy_thresh: float = 0.25,
        freq_residual_scale: float = 0.5,
        head_scale_range: Tuple[float, float] = (0.5, 1.5),
    ):
        super().__init__()
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.norm_eval = norm_eval

        if freq_base_resolutions is None:
            # default for 224x224 input: 56, 28, 14, 7
            freq_base_resolutions = [(56, 56), (28, 28), (14, 14), (7, 7)]
        assert len(use_frequencies) == self.num_layers
        assert len(freq_base_resolutions) == self.num_layers

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=norm_layer if self.patch_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
                # NEW: freq cfg
                use_frequency=use_frequencies[i_layer],
                freq_base_resolution=freq_base_resolutions[i_layer],
                freq_energy_thresh=freq_energy_thresh,
                freq_residual_scale=freq_residual_scale,
                head_scale_range=head_scale_range,
            )
            self.layers.append(layer)

        self.extra_norms = nn.ModuleList([nn.LayerNorm(embed_dims[i]) for i in range(4)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except Exception:
                pass

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            if (load_checkpoint is None) or (get_root_logger is None):
                raise RuntimeError("mmcv/mmseg is required for loading checkpoints in this file.")
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x)
            if i in self.out_indices:
                x_out = self.extra_norms[i](x_out)
                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
