import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class Residual(nn.Module):
    def __init__(self, dim, norm, activ):

        super(Residual, self).__init__()
        self.model = []
        self.model += [RBlock( dim, 1, norm, activ)]
        self.model += [RBlock( dim, 1, norm, activ)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):

        R = x
#        out = self.model(x.view(x.size(0), -1))
        out = self.model(x)
        out += x

        return out


class RBlock(nn.Module):
    def __init__(self, dim, stride, norm, activation):
        super(RBlock, self).__init__()

        self.conv1=nn.Conv2d( in_channels=dim, out_channels=dim, kernel_size=3, stride=stride, padding=1, bias=False)
        norm_dim = dim

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None


    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = self.activation(out)
        return out
