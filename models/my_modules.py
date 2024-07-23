# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Block(nn.Module):
    def __init__(
            self, dim, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = Mamba(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

from diffusers import ModelMixin,ConfigMixin
from diffusers.configuration_utils import register_to_config
from typing import Callable, List, Optional, Union
class CusMamba(nn.Module):
    def __init__(self, dim, dropout=0., decoder=False):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for _ in range(8)])
        self.gate=nn.Linear(1024,1,bias=False)
        self.gate.weight.data.fill_(0)
        self.proj=nn.Linear(1024,1024)

    def forward(self, x):
        residual = None
        hidden_states = x

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
        hidden_states = residual

        return self.proj(hidden_states),self.gate(hidden_states)

class CusMamba123(nn.Module):
    def __init__(self, dim, dropout=0., decoder=False):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for _ in range(16)])
        self.gate=nn.Linear(1024,1,bias=False)
        self.gate.weight.data.fill_(0)
        self.proj=nn.Linear(1024,1024)

    def forward(self, x):
        residual = None
        hidden_states = x

        for i in range(len(self.layers) // 2):
            hidden_states_f, residual_f = self.layers[i * 2](
                hidden_states, residual,
            )
            hidden_states_b, residual_b = self.layers[i * 2 + 1](
                hidden_states.flip([1]), None if residual == None else residual.flip([1]),
            )
            hidden_states = hidden_states_f + hidden_states_b.flip([1])
            residual = residual_f + residual_b.flip([1])

        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states

        hidden_states = residual[:, -77:]

        return self.proj(hidden_states),self.gate(hidden_states)


from einops import rearrange

class MAdapterBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down: bool = False, magic=False):
        super().__init__()
        self.downsample = None
        if down:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.in_conv = None
        self.out_channels=out_channels
        if in_channels != out_channels:
            self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.layers=nn.ModuleList([Block(out_channels) for _ in range(4)])
        self.magic=magic
        if magic:
            self.global_local_mamba=nn.ModuleList([Block(out_channels) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This method takes tensor x as input and performs operations downsampling and convolutional layers if the
        self.downsample and self.in_conv properties of AdapterBlock model are specified. Then it applies a series of
        residual blocks to the input tensor.
        """
        _b, _n,  _c, _h, _w = x.size()

        x = rearrange(x, 'b n c h w->(b n) c h w')

        if self.downsample is not None:
            x = self.downsample(x)
            _h =_h//2
            _w= _w//2

        if self.in_conv is not None:
            x = self.in_conv(x)
            _c=self.out_channels
        x = rearrange(x, '(b n) c h w->(b n) (h w) c',b=_b, n=_n, c=_c, w=_w, h=_h)

        residual = None
        hidden_states = x
        for i in range(len(self.layers) // 2):
            hidden_states_f, residual_f = self.layers[i * 2](
                hidden_states, residual,
            )
            hidden_states_b, residual_b = self.layers[i * 2 + 1](
                hidden_states.flip([1]), None if residual == None else residual.flip([1]),
            )

            if self.magic:
                magic_hidden_states= rearrange(hidden_states, '(b n) (h w) c->(b h w) n c',b=_b,h=_h)
                magic_residual = rearrange(residual, '(b n) (h w) c->(b h w) n c', b=_b, h=_h) if residual is not None else None
                hidden_states_n, residual_n = self.global_local_mamba[i](magic_hidden_states,None if residual == None else magic_residual)
                hidden_states_n = rearrange(hidden_states_n, '(b h w) n c->(b n) (h w) c', b=_b,n=_n,c=_c,w=_w, h=_h)
                residual_n = rearrange(residual_n, '(b h w) n c->(b n) (h w) c', b=_b, n=_n, c=_c, w=_w, h=_h)

            hidden_states = hidden_states_f + hidden_states_b.flip([1]) if not self.magic else hidden_states_f + hidden_states_b.flip([1]) + hidden_states_n

            residual = residual_f + residual_b.flip([1]) if not self.magic else residual_f + residual_b.flip([1]) + residual_n
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
        x = residual
        x = rearrange(x, '(b n) (h w) c->b n c h w',b=_b,h=_h)
        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class MambaAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 5,
        channels: List[int] = [320, 640, 1280, 1280],
        downscale_factor: int = 8,
    ):
        super().__init__()
        in_channels = in_channels * downscale_factor**2
        # self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        # self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        num_patches=64*64
        self.patch_embed=nn.Sequential(
            nn.Conv2d(5,80,3,2,1), LayerNorm(80), nn.SiLU(),
            nn.Conv2d(80,160,3,2,1), LayerNorm(160), nn.SiLU(),
            nn.Conv2d(160,320,3,2,1))


        #self.pos_embed = nn.Parameter(torch.zeros(1, 1, channels[0], 64, 64))
        self.image_embed = nn.Parameter(torch.zeros(1, 7, channels[0], 1, 1))
        #trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.image_embed, std=.02)

        self.block1 = MAdapterBlock(in_channels, channels[0], magic=False)
        self.block2 = MAdapterBlock(channels[0], channels[1], down=True, magic=True)
        self.block3 = MAdapterBlock(channels[1], channels[2], down=True, magic=True)
        self.block4 = MAdapterBlock(channels[2], channels[3], down=True, magic=True)

        self.total_downscale_factor = downscale_factor * 2 ** (len(channels) - 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, n ,c, h, w = x.size()

        x = rearrange(x, 'b n c h w->(b n) c h w')
        #x = self.unshuffle(x)
        x = self.patch_embed(x)
        x = rearrange(x, '(b n) c h w->b n c h w', b=b)#torch.Size([1, 7, 320, 64, 64])
        x = x + self.image_embed #+ self.pos_embed

        feat1 = self.block1(x)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)
        feat4 = self.block4(feat3)
        return [feat1[:,-1], feat2[:,-1], feat3[:,-1], feat4[:,-1]]


if __name__ == '__main__':
    a=MambaAdapter()
    a.cuda()
    inp=torch.randn(1, 6, 5, 512, 512).cuda()
    out=a(inp)
    print(out)
    for o in out:
        print(o.size())

    d= CusMamba(1024).cuda()
    inputd= torch.randn(1, 84, 1024).cuda()
    print(d(inputd).size())