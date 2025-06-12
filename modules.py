import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
# from ..modules.conv import Conv, DWConv, RepConv, autopad

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class SIFA(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride=1, reduction=32):
        super().__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(inp, inp * (kernel_size**2), kernel_size, padding=kernel_size//2,
                                                stride=stride, groups=inp,
                                                bias=False),
                                      nn.BatchNorm2d(inp * (kernel_size**2)),
                                      nn.ReLU()
                                      )
        
        self.G = 8
        self.channel = inp
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(self.channel // (self.G), self.channel // (self.G))
        self.cweight = Parameter(torch.zeros(1, self.channel // (self.G), 1, 1))
        self.cbias = Parameter(torch.ones(1, self.channel // (self.G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, self.channel // (self.G), 1, 1))
        self.sbias = Parameter(torch.ones(1, self.channel // (self.G), 1, 1))
        self.weights_add = Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()

        self.conv = Conv(inp, oup, k=kernel_size, s=kernel_size, p=0)


    def forward(self, x):
        b, c = x.shape[0:2]
        generate_feature = self.generate(x)  # b, ck^2, h, w
        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size**2, h, w)  # b, c, k^2, h, w
        
        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)
        b, c, h, w = generate_feature.size()
        # group into subfeatures
        generate_feature = generate_feature.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel attention
        x_channel = self.avg_pool(generate_feature)  # bs*G,c//(G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(G),1,1
        x_channel = generate_feature * self.sigmoid(x_channel)  # bs*G,c//(G),h,w

        # spatial attention
        x_spatial = self.gn(generate_feature)  # bs*G,c//(G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(G),1,1 × bs*G,c//(G),h,w = bs*G,c//(G),h,w
        x_spatial = generate_feature * self.sigmoid(x_spatial)  # bs*G,c//(G),h,w

        # concatenate along channel axis
        out = x_channel* self.weights_add + (1-self.weights_add)*x_spatial # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # output
        return self.conv(out)
    
if __name__ == "__main__":

    x = torch.randn(1, 64, 640, 640)
    module = SIFA(64, 64, 3, 2)
    out = module(x)
    print(x.shape)