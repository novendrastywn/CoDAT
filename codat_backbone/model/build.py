
'''
Build the SHViT model family
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from .codat import CoDAT_Backbone

@register_model
def CoDAT_S(pretrained=False, pretrained_cfg=None, **kwargs):
    model = CoDAT_Backbone(
        embed_dims=[192, 384, 448],
        depths=[1, 2, 2],
        att_ratio=[0.25, 0.25, 0.25], 
        sr_ratio=[2, 2, 1],
        drop_rate=0.05,
        **kwargs)
    return model

@register_model
def CoDAT_M(pretrained=False, pretrained_cfg=None, **kwargs):
    model = CoDAT_Backbone(
        embed_dims=[200, 384, 448],
        depths=[2, 4, 4],
        att_ratio=[0.25, 0.25, 0.25], 
        sr_ratio=[2, 2, 1],
        drop_rate=0.1,
        **kwargs)
    return model

@register_model
def CoDAT_L(pretrained=False, pretrained_cfg=None, **kwargs):
    model = CoDAT_Backbone(
        embed_dims=[200, 384, 448],
        depths=[2, 4, 4],
        att_ratio=[0.25, 0.25, 0.25], 
        sr_ratio=[2, 2, 1],
        drop_rate=0.1,
        **kwargs)
    return model


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

def reparameterize(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'reparam'):
            reparametrized = child.reparam()
            setattr(net, child_name, reparametrized)
            reparameterize(reparametrized)
        elif hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        else:
            reparameterize(child)
    
    return net