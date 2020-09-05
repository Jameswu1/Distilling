import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class Residual(nn.Module):
    def __init__(self, dim, n_blk, norm, activ):

        super(FCN, self).__init__()
        self.model = []
        self.model += [RBlock( dim, dim, 1, norm, activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock( dim, dim, 1, norm, activ)]
        self.model = nn.Sequential(*self.model)
        self.conv2=nn.Conv2d( dim, dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d( dim )
        

        def forward(self, x):
            R = x
            out = self.model(x.view(x.size(0), -1))
            out += x
            return out


class RBlock(nn.Module):
    def __init__(self,i_channel,o_channel,stride=1,norm,activation):
        super(RBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channels=i_channel,out_channels=o_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
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
