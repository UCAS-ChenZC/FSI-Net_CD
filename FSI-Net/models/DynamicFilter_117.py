import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight

class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=None, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=False,
                 # ===== 新增：低频偏置相关 =====
                 lowfreq_bias=True,
                 lowfreq_sigma=0.35,   # 越大：低频范围越宽；越小：只强调很靠近DC
                 lowfreq_sharp=1.0,    # mask幂指数，>1 更“硬”，<1 更“软”
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize

        # activation
        if act1_layer is None:
            # 如果你想更偏低频，建议先把 StarReLU 换成 GELU（可选）
            act1_layer = nn.GELU
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)

        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2, dtype=torch.float32) * 0.02
        )
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

        # ===== 新增：低频mask（register_buffer 不参与梯度）=====
        self.lowfreq_bias = lowfreq_bias
        self.lowfreq_sigma = float(lowfreq_sigma)
        self.lowfreq_sharp = float(lowfreq_sharp)

        lp = self._build_lowpass_mask(self.size, self.filter_size, sigma=self.lowfreq_sigma)  # (H, W/2+1)
        self.register_buffer("lowpass_mask", lp)  # float32

    @staticmethod
    def _build_lowpass_mask(h: int, w_half: int, sigma: float = 0.35):
        """
        构建 rfft2 频域下的低通mask（未shift，左上角是低频/DC）
        返回 shape = (h, w_half)，值域 [0,1]，低频接近1，高频接近0
        """
        # 归一化频率坐标：ky in [0,1], kx in [0,1]
        ky = torch.linspace(0.0, 1.0, steps=h)
        kx = torch.linspace(0.0, 1.0, steps=w_half)
        yy, xx = torch.meshgrid(ky, kx, indexing="ij")
        r = torch.sqrt(yy ** 2 + xx ** 2)

        # 高斯低通（你也可以换成 1-r 的线性mask）
        mask = torch.exp(-(r ** 2) / (2 * (sigma ** 2) + 1e-12))
        mask = mask / (mask.max() + 1e-12)
        return mask.to(torch.float32)

    def forward(self, x):
        B, H, W, _ = x.shape

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters, -1).softmax(dim=1)

        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  # (B, H, W/2+1, C) complex

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1], x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)

        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)  # (B,H,W/2+1,C) complex

        # 统一 reshape（你原始逻辑保留）
        if not self.weight_resize:
            # 若不 resize，这里假设 self.size == H 且 self.filter_size == W/2+1
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        else:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)

        # ===== 核心修改：低频门控，让模型主要在低频学习 =====
        if self.lowfreq_bias:
            mask = self.lowpass_mask  # (h, w_half)
            if self.weight_resize:
                # mask resize 到当前频谱尺寸
                mask = F.interpolate(mask[None, None, :, :],
                                     size=(x.shape[1], x.shape[2]),
                                     mode="bilinear",
                                     align_corners=False)[0, 0]
            # 更“硬”的低频选择：mask^sharp
            if self.lowfreq_sharp != 1.0:
                mask = mask.clamp(0, 1) ** self.lowfreq_sharp

            # broadcast 到 (B,H,W/2+1,C)
            mask = mask.to(weight.device).to(weight.real.dtype).unsqueeze(0).unsqueeze(-1)

            # 高频处 weight -> 1（恒等），低频处 weight 保持可学习
            one = torch.ones_like(weight)
            weight = one + mask * (weight - one)

        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)
        return x
    
if __name__ == '__main__':
    block = DynamicFilter(32, size=64) # size==H,W
    input = torch.rand(3, 64, 64, 32) #输入 B C H W
    output = block(input)
    print(input.size())
    print(output.size())