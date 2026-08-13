import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from einops import rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers import SqueezeExcite
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

class Conv2d_BN(nn.Sequential):
    def __init__(self, inc, ouc, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            inc, ouc, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(ouc))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def reparam(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:], 
                            stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, 
                            groups=self.c.groups, device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, dim, **kwargs):
        super().__init__(1, dim, **kwargs)

class BN_Linear(nn.Sequential):
    def __init__(self, inc, ouc, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(inc))
        self.add_module('l', torch.nn.Linear(inc, ouc, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def reparam(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(nn.Module):

    def __init__(self, m, dim, drop=0., ls_init=0):

        super().__init__()
        self.m = m
        self.drop = drop
        # self.ls = nn.Parameter(ls_init*torch.ones(dim, 1, 1), requires_grad=True)

        if  self.drop > 0:
            self.forward = self.forward_drop
            self.drop_path = DropPath(drop)
        else:
            self.forward = self.forward_deploy

    def forward_drop(self, x):
        return x + self.drop_path(self.m(x))  # torch.rand(x.size(0), 1, 1, 1,
                                         # device=x.device).ge_(self.drop).div(1 - self.drop).detach()

    def forward_deploy(self, x):
        return x + self.m(x)

        
    @torch.no_grad()
    def reparam(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.reparam()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m

        else:
            return self

class SSEmodule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 1, 1)
    
    def forward(self, x):
        gca = F.adaptive_avg_pool2d(x, 1)
        exc = F.sigmoid(self.conv(gca))
        return exc*x

class StrideConv(nn.Module):
    def __init__(self, inc, ouc, ks=3, s=16, act_layer=nn.SiLU):
        super().__init__()
        pad=0 if (ks % 2)==0 else ks//2
        blocks = 1 if s==2 else math.ceil(s**0.5) 
        dims = [inc] + [x.item() for x in ouc//2**torch.arange(blocks-1, -1, -1)]
        stem = [nn.Sequential(
                Conv2d_BN(dims[i], dims[i+1], ks=ks, stride=2, pad=pad),
                act_layer() if i < (blocks-1) else nn.Identity())
                for i in range (blocks)]
        self.stem = nn.Sequential(*stem)
        
    def forward(self, x):
        return self.stem(x)

class PatchEmbedding(nn.Module):
    """ Channel Attention Pacth Embedding"""
    def __init__(self, inc=3, ouc=768, ks=3, s=4, se=0, act_layer=nn.SiLU):
        super().__init__()

        if s>2:
            self.conv_proj = StrideConv(inc, ouc, ks=ks, s=s, act_layer=act_layer)
        else:
            hid_dim = int(inc * 4)
            self.conv_proj = nn.Sequential(
                            Conv2d_BN(inc, hid_dim, 1, 1, 0, ),
                            act_layer(),
                            Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim,), 
                            Conv2d_BN(hid_dim, ouc, 1, 1, 0,))

        self.se = SSEmodule(ouc) if se !=0 else nn.Identity()
    def forward(self, x):
        x = self.conv_proj(x) # Convolutional Patch Embedding
        x = self.se(x)
        return x 

class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def reparam(self):
        classifier = self.classifier.reparam()
        if self.distillation:
            classifier_dist = self.classifier_dist.reparam()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier
    
class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=nn.GELU):
        super().__init__()
        hidden_features = min(int(mlp_ratio * dim), 1280)
        self.fc1 = BN_Linear(dim, hidden_features)
        self.act = act_layer()
        self.fc2 = BN_Linear(hidden_features, num_classes)
  
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class ConvFFN(nn.Module):
    def __init__(self, inc, hidd=None, ouc=None, act_layer=nn.SiLU):
        super().__init__()
        ouc = ouc or inc
        hidd = hidd or inc
        self.dwc = Conv2d_BN(inc, inc, 3, 1, 1, groups=inc)
        self.ffn = nn.Sequential(
                    Conv2d_BN(inc, hidd, 1, 1),
                    act_layer(),
                    Conv2d_BN(hidd, ouc,1, 1))
        
    def forward(self, x):
        x = self.dwc(x)
        x = self.ffn(x)
        return x

class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class MHSA(nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4:
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
        qkv = (
            self.qkv(self.norm(x))
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class LocalAggregation(nn.Module):
    def __init__(self, dim, ks=3):
        super().__init__()
        pad = ks//2
        self.conv = nn.Conv2d(dim, dim, ks, 1, pad, groups=dim)
        self.repconv = nn.Conv2d(dim, dim, ks//2, 1, pad//2, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
    
    def forward(self, x):
        xr = self.conv(x) + self.repconv(x) 
        return self.bn(xr)
    
    @torch.no_grad()
    def reparam(self):
        conv = self.conv

        repconv=self.repconv; self.__delattr__('repconv')
        kw, kh = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                     (conv.weight.shape[3]-repconv.weight.shape[3])//2
        conv_w = conv.weight + F.pad(repconv.weight, [kh,kh,kw,kw])
        conv_b = conv.bias + repconv.bias          

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv_w * w[:, None, None, None]
        b = bn.bias + (conv_b - bn.running_mean) * bn.weight / \
                    (bn.running_var + bn.eps)**0.5
        self.__delattr__('bn')

        dim = conv.in_channels
        ks = conv.kernel_size
        pad = conv.padding
        m = nn.Conv2d(dim, dim, ks, 1, pad, groups=dim)

        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        self.__delattr__('conv')
        return m

class SCAtt(nn.Module):

    def __init__(self, ks=7):
        super().__init__()
        assert ks in {3, 7}, "kernel size must be 3 or 7"
        pad = 3 if ks == 7 else 1
        self.conv = nn.Conv2d(2, 1, ks, padding=pad, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        att = torch.cat([torch.mean(x, 1, keepdim=True), 
                         torch.max(x, 1, keepdim=True)[0]], 1)
        return x * self.act(self.conv(att))

class SSHAtt(nn.Module):
    """Sparse Single Head Attention"""
    def __init__(self, dim, att_ratio=0.25, sr_ratio=1, qkdim=16, **kwargs):
        super().__init__()
        self.scale = qkdim**-0.5
        self.att_dim = int (dim * att_ratio)
        self.split_index = (qkdim, qkdim, self.att_dim)
        # self.sparse_proj = nn.Sequential(
        #                     nn.AvgPool2d(3, stride=sr_ratio, padding=1),
        #                     Conv2d_BN(dim, 2*qkdim + self.att_dim, 1, 1))
        self.sparse_proj = Conv2d_BN(dim, 2*qkdim + self.att_dim, 1, sr_ratio)
        
        if sr_ratio>1: 
            self.local_prop  = nn.ConvTranspose2d(self.att_dim, self.att_dim, 
                                                  sr_ratio, sr_ratio, groups=self.att_dim)
        else:
            self.local_prop = nn.Identity()

    def forward(self, x):
        
        x = self.sparse_proj(x)
        B, C, H, W = x.shape
        q, k, v = torch.split(x, self.split_index, dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        attn = (v @ attn.transpose(-2, -1)).reshape(B, self.att_dim, H, W)

        return self.local_prop(attn)

class CoDA(nn.Module):
    def __init__(self, dim, att_ratio=0.25, sr_ratio=1, **kwargs):
        super().__init__()
        self.att_dim = int(dim * att_ratio)
        self.loa  = LocalAggregation(dim, 3)
        self.csa  = SCAtt(ks=3)
        # self.csa  = nn.Identity()
        self.ssha = SSHAtt(dim, att_ratio, sr_ratio)
        self.proj = Conv2d_BN(self.att_dim + dim, dim)
        # self.proj = Conv2d_BN(dim, dim)
        # self.act = act_layer()

    def forward(self, x):
        x   = self.loa(x)
        att = self.ssha(x)
        ctt = self.csa(x)
        # print(att.shape, ctt.shape)
        x = self.proj(torch.cat((att, ctt), dim=1))
        # x = self.proj(ctt)
        return x

class CoDATBlock(nn.Module):

    def __init__(self, dim, att_ratio=0.25, sr_ratio=1, act_layer=nn.SiLU, drop=0.):
        super().__init__()           
        hid = int(dim * 2)
        ffn_pra = ConvFFN(inc=dim, hidd=hid, act_layer=act_layer)
        ffn_pre = ConvFFN(inc=dim, hidd=hid, act_layer=act_layer)
        token_mix = CoDA(dim, att_ratio, sr_ratio)
        # token_mix = MDTA(dim)
        # token_mix = MHSA(dim, proj_drop=drop)
        self.ffn_pre = Residual(ffn_pre, dim, drop, 1)
        self.token_mix = Residual(token_mix, dim, drop, 1)
        self.ffn_pra = Residual(ffn_pra, dim, drop, 1)

    def forward(self, x):
        return self.ffn_pra(self.token_mix(self.ffn_pre(x)))

    def shifting(self, num_segments=8, shift_div=8):
        """Shift the first conv of the block.

        Args:
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift.
        """
        # self.ffn_pre.m = TemporalShift(
        #     self.ffn_pre.m, num_segments=num_segments, shift_div=shift_div)
        # self.token_mix.m = TemporalShift(
        #     self.token_mix.m, num_segments=num_segments, shift_div=shift_div)
        self.ffn_pra.m = TemporalShiftAttention(
            self.ffn_pra.m, num_segments=num_segments, shift_div=shift_div)

        return self

class CoDAT_Backbone(nn.Module):
    def __init__(self, img_size=224, num_classes=1000, embed_dims=[128, 384, 512], embed_stride=[16, 2, 2], 
                 depths=[3, 9, 3], att_ratio=[0.25, 0.25, 0.25], sr_ratio=[8, 2, 2], num_stages=3, 
                 drop_rate=0.05, cape=[1, 1, 1], act_layer=nn.SiLU, distillation=False,
                 head_dropout=0.0, head_init_scale=1.0, pre_head='avg', pretrained=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.downsamplings=nn.ModuleList()
        self.stages=nn.ModuleList()
        for i in range(num_stages):
            patch_embed = PatchEmbedding(3 if i == 0 else embed_dims[i - 1], embed_dims[i],
                                         2 if embed_stride[i] == 3 else 3, embed_stride[i], 
                                         cape[i], act_layer)
            self.downsamplings.append(patch_embed) 
            
            block = nn.Sequential(
                *[CoDATBlock(dim=embed_dims[i],
                                 att_ratio=att_ratio[i], 
                                 sr_ratio=sr_ratio[i],  
                                 drop=dpr[cur + j], 
                                 act_layer=act_layer)
                for j in range(depths[i])])
            self.stages.append(block)

            cur += depths[i]

        

        # classification head
        self.head = Classfier(embed_dims[-1], num_classes, distillation)

        if pre_head is "avg":
            self.pre_head = nn.AdaptiveAvgPool2d(1)
        elif pre_head is "gdc":
            import math
            ks = 4 if math.prod(embed_stride) == 64 else 7
            self.pre_head = Conv2d_BN(embed_dims[-1], embed_dims[-1], 4, 1, groups=embed_dims[-1])
        else:
            raise ValueError("pre_head must be 'avg' or 'gd'")


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained):
        print("Load Backbone Pretrained weight")
        checkpoint = torch.load(pretrained, weights_only=False, map_location='cpu')
        d = checkpoint['model']
            # D = self.state_dict()
            # for k in d.keys():
            #     if D[k].shape != d[k].shape:
            #         d[k] = d[k][:, :, None, None]
        self.load_state_dict(d)


    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            x = self.downsamplings[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class TemporalShift(nn.Module):
    """Temporal shift module.

    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_

    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """

    def __init__(self, net, num_segments=3, shift_div=8):
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.shift_div = shift_div
        print(f'Temporal Shift with div {shift_div} and {num_segments} segments.')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.shift(x, self.num_segments, shift_div=self.shift_div)
        return self.net(x)

    @staticmethod
    def shift(x, num_segments, shift_div=3):
        """Perform temporal shift operation on the feature.

        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.

        Returns:
            torch.Tensor: The shifted feature.
        """
        # [N, C, H, W]
        n, c, h, w = x.size()

        # [N // num_segments, num_segments, C, H*W]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = x.view(-1, num_segments, c, h * w)

        # get shift fold
        fold = c // shift_div

        # split c channel into three parts:
        # left_split, mid_split, right_split
        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold:2 * fold, :]
        right_split = x[:, :, 2 * fold:, :]

        # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
        # because array on caffe inference must be got by computing

        # shift left on num_segments channel in `left_split`
        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)

        # shift right on num_segments channel in `mid_split`
        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)

        # right_split: no shift

        # concatenate
        out = torch.cat((left_split, mid_split, right_split), 2)

        # [N, C, H, W]
        # restore the original dimension
        return out.view(n, c, h, w)

class TemporalShiftAttention(nn.Module):
    """Temporal shift module.

    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_

    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """

    def __init__(self, net, num_segments=3, shift_div=8):
        super().__init__()
        self.net = net
        self.n_seg = num_segments
        self.shift_div = shift_div
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cvg = nn.Conv2d(num_segments, 1, 1, 1, bias=True)
        self.cvl = nn.Conv2d(num_segments, num_segments, 1, 1, bias=True)
        self.cvm = nn.Conv2d(num_segments, num_segments, 1, 1, bias=True)
        print(f'Temporal Shift with div {shift_div} and {num_segments} segments.')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.gated_shift(x, self.gap, 
                            self.cvg, self.cvl, self.cvm,  
                            self.n_seg, self.shift_div)
        return self.net(x)

    @staticmethod
    def gated_shift(x, adapool, 
                    cvg, cvl, cvm, 
                    n_seg, shift_div=3):
        """Perform temporal shift operation on the feature.

        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.

        Returns:
            torch.Tensor: The shifted feature.
        """
        # split c channel into three parts:
        # left_split, mid_split, right_split
        n, c, h, w = x.size()
        x = x.view(-1, n_seg, c, h * w)
        fold = c // shift_div

        gap = adapool(x)
        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold:2 * fold, :]
        right_split = x[:, :, 2 * fold:, :]
        att = mid_split[:, :1, :, :] * F.sigmoid(cvg(gap))
        # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
        # because array on caffe inference must be got by computing

        # shift left on num_segments channel in `left_split`
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, att), 1) * F.sigmoid(cvl(gap))

        # shift right on num_segments channel in `mid_split`
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((att, mid_split), 1) * F.sigmoid(cvm(gap))
        # right_split: no shift

        # concatenate
        return torch.cat((left_split, mid_split, right_split), 
                         2).view(n, c, h, w)

def add_temporal_shift(net, num_segments, shift_div):
    for child_name, child in net.named_children():
        if hasattr(child, 'shifting'):
            shifted = child.shifting(num_segments, shift_div)
            setattr(net, child_name, shifted)
            add_temporal_shift(shifted, num_segments, shift_div)
        else:
            add_temporal_shift(child, num_segments, shift_div)
    return net

def reparameterize(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'reparam'):
            reparametrized = child.reparam()
            setattr(net, child_name, reparametrized)
            reparameterize(reparametrized)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            reparameterize(child)
    
    return net


class CoDAT(CoDAT_Backbone):

    arch_zoo = {
        'codat_s':
            dict(embed_dims=[192, 384, 448],
                 depths=[1, 2, 2],
                 att_ratio=[0.25, 0.25, 0.25], 
                 sr_ratio=[2, 2, 1],
                 ),
        'codat_m':
            dict(embed_dims=[200, 384, 448], #200, 384, 448
                 depths=[2, 4, 4],
                 att_ratio=[0.25, 0.25, 0.25], 
                 sr_ratio=[2, 2, 1],),
        'codat_l':
            dict(embed_dims=[280, 448, 512],
                depths=[5, 5, 4],
                att_ratio=[0.25, 0.25, 0.25], 
                sr_ratio=[4, 2, 1],)
        }

    def __init__(self,
                 num_classes=400,
                 num_segments=8,
                 is_shift=True,
                 shift_div=8,
                 pretrained2d=True,
                 arch="parformer_s2",
                 drop_path_rate: float = 0.05,
                 fc_drop_rate: float = 0.05,
                 init_std: float = 0.001,
                 embedding_head=True,
                 **kwargs):
        if isinstance(arch, str):
            assert arch in self.arch_zoo, f'"arch": "{arch}"' \
                f' is not one of the {list(self.arch_zoo.keys())}'
            arch = self.arch_zoo[arch]
        elif not isinstance(arch, dict):
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')
        super().__init__(
                    embed_dims=arch["embed_dims"],
                    depths=arch["depths"],
                    att_ratio=arch["att_ratio"], 
                    sr_ratio=arch["sr_ratio"],
                    pretrained=pretrained2d,
                    drop_rate=drop_path_rate 
        )
        super().init_weights(pretrained2d)
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.pretrained2d = pretrained2d
        self.init_structure()
        self.__delattr__('pre_head') #delete 2d head
        self.__delattr__('head')#delete 2d head
        self.fc_drop_rate = fc_drop_rate
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.num_classes = num_classes

        self.dropout = nn.Dropout(p=self.fc_drop_rate )

        self.fc_cls = nn.Sequential(
                      nn.Linear(arch["embed_dims"][-1], 1280),
                      nn.SiLU(),
                      nn.Linear(1280, self.num_classes),
                      )
        # self.fc_cls = nn.Linear(self.in_channels, self.num_classes)     

        if embedding_head:
            self.fc_emb  = nn.Sequential(
                                nn.Linear(arch["embed_dims"][-1], 600),
                                nn.Tanh(),
                                nn.Linear(600,300))
            self.forward = self.forward_emb

        else:    
            self.forward = self.forward_cls

        self.spt_pool = nn.AdaptiveAvgPool2d(1)

        self._init_weights(self.fc_cls)

    def init_structure(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if self.is_shift:
            add_temporal_shift(self, self.num_segments, self.shift_div)

    def switch_to_deploy(self):
        
        return reparameterize(self)
    
    def forward_cls(self, x):
        """unpack tuple result."""
        x = super().forward(x) # [N * num_segs, in_channels, H/64, W/64]
        # print(x.shape)
        x = self.spt_pool(x).flatten(1) # [N * num_segs, in_channels]
        x = self.dropout(x)
        x = self.fc_cls(x)
        x = x.view((-1, self.num_segments) + x.size()[1:])
        return x.mean(dim=1, keepdim=True).squeeze(1) # [N, num_classes]

    def forward_emb(self, x):
        """unpack tuple result."""
        x = super().forward(x) # [N * num_segs, in_channels, H/64, W/64]
        # print(x.shape)
        x = self.spt_pool(x).flatten(1) # [N * num_segs, in_channels]
        x = self.dropout(x)
        cls_score = self.fc_cls(x)
        emb_score = self.fc_emb(x)
        cls_score = cls_score.view((-1, self.num_segments) + cls_score.size()[1:])
        cls_score = cls_score.mean(dim=1, keepdim=True).squeeze(1) # [N, num_classes]
        emb_score = emb_score.view((-1, self.num_segments) + emb_score.size()[1:])
        emb_score = emb_score.mean(dim=1, keepdim=True).squeeze(1) # [N, num_classes]
        return cls_score, emb_score
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

@register_model
def codat_act_s(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = CoDAT(
        arch="codat_s",
        pretrained2d="./models/parformer_s3_checkpoint.pth",
        embedding_head=False,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def codat_act_m(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = CoDAT(
        arch="codat_m",
        pretrained2d="./models/parformer_s2_checkpoint.pth",
        embedding_head=False,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def codat_act_l(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = CoDAT(
        arch="codat_l",
        pretrained2d="./models/parformer_s2_checkpoint.pth",
        embedding_head=False,
        **kwargs)
    model.default_cfg = _cfg()
    return model


