import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.utils as KU
class SpatialTransformer(nn.Module):
    def __init__(self, h,w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h,w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, padding_mode='zeros', align_corners=False)

class Conv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, dilation=1, norm=None, act=nn.LeakyReLU,bias=False):
        super(Conv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=bias, dilation=dilation)]
        if not norm is None:
            model += [norm(n_out, affine=False)]
        if act is nn.LeakyReLU:
            model += [act(negative_slope=0.1,inplace=True)]
        elif act is None:
            model +=[]
        else:
            model +=[act()]
        self.model = nn.Sequential(*model)
        # elif == 'Group'

    def forward(self, x):
        return self.model(x)

class ResConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, dilation=1, norm=None,):
        super(ResConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False, dilation=dilation)]
        if not norm is None:
            model += [norm(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        # elif == 'Group'

    def forward(self, x):
        return self.model(x)+x

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, norm=None, activation='LReLU', kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)]
        if norm == 'Batch':
            model += [nn.BatchNorm2d(out_channels)]
        if activation == 'LReLU': ## 默认使用LeakyReLU作为激活函数
            model += [nn.LeakyReLU(inplace=True)]
        elif activation == 'Sigmoid':
            model += [nn.Sigmoid()]
        elif activation == 'ReLU':
            model += [nn.ReLU()]
        elif activation == 'Tanh':
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)