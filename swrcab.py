import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class RIR(nn.Module):
    def __init__(self, dim, n_blk, norm, activ):

        super(FCN, self).__init__()

        self.con1 = nn.Conv2d( dim, dim, kernal_size=3, stride=1, padding=1, bias = False)
        self.con2 = nn.Conv2d( dim, dim, kernal_size=3, stride=1, padding=1, bias = False)
        self.con3 = nn.Conv2d( dim, dim, kernal_size=3, stride=1, padding=1, bias = False)

        self.activ = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(dim)
        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d()


        def forward(self, x):

            out = self.con1(x.view(x.size(0), -1),x.view(x.size(0),-1))
            out = self.activ(out)
            out1 = out
            out = self.con2(out)
            out1 = self.con3(out)
            out1 = self.sigmoid(out1)
            out1 = out * out1
            out1 = self.avg(out1)





            out += x
            return out


