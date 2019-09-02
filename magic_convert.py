import numpy as np
import torch
import numpy as np
from collections import OrderedDict
b=torch.load('./out/mobile_model_epoch_1.pt')
# a=torch.load('network.torch')
# key_a=a.keys()
# key_b=b.keys()
# result=OrderedDict()
# for ka in key_a:
#     for kb in key_b:
#         if(ka in kb):
#             result[kb]=a[ka]
# print(len(result.keys()))

# torch.save(result,"pretrained.torch")

c=torch.load("pretrained.torch")
print(b.keys())


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import datetime
from collections import OrderedDict

def Conv_3x3(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6()
    )

def Conv_1x1(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6()
    )

def SepConv_3x3(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class MBConv3_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride):  
        super(MBConv3_3x3, self).__init__()
        mid_channels = int(3 * in_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(),

            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(),

            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.use_skip_connect = (1 == stride and in_channels == out_channels)

    def forward(self, x):
        if self.use_skip_connect:
            return self.block(x) + x
        else:
            return self.block(x)    

class MBConv3_5x5(nn.Module):
    def __init__(self, in_channels, out_channels, stride):  
        super(MBConv3_5x5, self).__init__()
        mid_channels = int(3 * in_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(),

            nn.Conv2d(mid_channels, mid_channels, 5, stride, 2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(),

            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.use_skip_connect = (1 == stride and in_channels == out_channels)

    def forward(self, x):
        if self.use_skip_connect:
            return self.block(x) + x
        else:
            return self.block(x) 

class MBConv6_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride):  
        super(MBConv6_3x3, self).__init__()
        mid_channels = int(6 * in_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(),

            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(),

            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.use_skip_connect = (1 == stride and in_channels == out_channels)

    def forward(self, x):
        if self.use_skip_connect:
            return self.block(x) + x
        else:
            return self.block(x) 

class MBConv6_5x5(nn.Module):
    def __init__(self, in_channels, out_channels, stride):  
        super(MBConv6_5x5, self).__init__()
        mid_channels = int(6 * in_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(),

            nn.Conv2d(mid_channels, mid_channels, 5, stride, 2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(),

            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.use_skip_connect = (1 == stride and in_channels == out_channels)

    def forward(self, x):
        if self.use_skip_connect:
            return self.block(x) + x
        else:
            return self.block(x) 

class MnasNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MnasNet, self).__init__()

        self.out_channels = int(1280 * width_mult)

        self.conv1 = Conv_3x3(3, int(32 * width_mult), 2)
        self.conv2 = SepConv_3x3(int(32 * width_mult), int(16 * width_mult), 1)

        self.feature = nn.Sequential(
            self._make_layer(MBConv3_3x3, 3, int(16 * width_mult), int(24 * width_mult), 2),
            self._make_layer(MBConv3_5x5, 3, int(24 * width_mult), int(64 * width_mult), 2)
        )
        self.feature1=nn.Sequential(
            self._make_layer(MBConv6_5x5, 3, int(64 * width_mult), int(80 * width_mult), 2),
            
        )
        self.feature2=nn.Sequential(
            self._make_layer(MBConv6_3x3, 2, int(80 * width_mult), int(128 * width_mult), 1)
        )
        self.feature3=nn.Sequential(
            self._make_layer(MBConv6_5x5, 4, int(128 * width_mult), int(192 * width_mult), 2)
        )
        self.feature4=nn.Sequential(
            self._make_layer(MBConv6_3x3, 1, int(192 * width_mult), int(256 * width_mult), 1)
        )

        # self.conv3 = Conv_1x1(int(256 * width_mult), int(1280 * width_mult), 1)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(int(1280 * width_mult), num_classes)

        self._initialize_weights()

    def _make_layer(self, block, blocks, in_channels, out_channels, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for _stride in strides:
            layers.append(block(in_channels, out_channels, _stride))
            in_channels = out_channels
        
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_() 

    def forward(self, x):
        result=OrderedDict()
        x = self.conv2(self.conv1(x))
        x1 = self.feature(x)
        result[1]=x1
        x=self.feature1(x1)
        x2=self.feature2(x)
        result[2]=x2
        x=self.feature3(x2)
        x3=self.feature4(x)
        result[3]=x3
        return result

if __name__ == '__main__':
    net = MnasNet()
    x = torch.randn(1,3,320,320)
    net(x)
    # for i in range(15):
    #     time1 = datetime.datetime.now()
    #     y = net(x)
    #     print('Time Cost: ', (datetime.datetime.now() - time1).microseconds)
    #y = net(x)
    #print(y)