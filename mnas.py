from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class list_(object):
#     def __init__(self,li=None):
#         if(li!=None):
#             self.li=[]
#         else:
#             self.li=li
#     def ret(self):
#         return self.li

result_list=[]
last_fm_list=[]
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
        mid_channels = int(6 * in_channels/1.125)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 5, stride, 2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.use_skip_connect = (1 == stride and in_channels == out_channels)

    def forward(self, x):
        if self.use_skip_connect:
            x1=self.block1(x)
            x1=self.block2(x1)
            last_fm_list.append(x1)
            x1=self.block3(x1)
            
            return x1 + x
        else:
            x1=self.block1(x)
            result_list.append(x1)
            x1=self.block2(x1)
            
            
            x1=self.block3(x1)
            
            return x1
class MnasNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MnasNet, self).__init__()

        self.out_channels = int(1280 * width_mult)

        self.conv1 = Conv_3x3(3, int(32 * width_mult), 2)
        self.conv2 = SepConv_3x3(int(32 * width_mult), int(16 * width_mult), 1)

        self.feature = nn.Sequential(
            self._make_layer(MBConv3_3x3, 3, int(16 * width_mult), int(24 * width_mult), 2),
            self._make_layer(MBConv3_5x5, 3, int(24 * width_mult), int(48 * width_mult), 2),
            self._make_layer(MBConv6_5x5, 3, int(48 * width_mult), int(80 * width_mult), 2),
            self._make_layer(MBConv6_3x3, 2, int(80 * width_mult), int(96 * width_mult), 1),
            self._make_layer(MBConv6_5x5, 4, int(96 * width_mult), int(192 * width_mult), 2)
            # self._make_layer(MBConv6_3x3, 1, int(192 * width_mult), int(320 * width_mult), 1)
        )


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
        # global result_list
        x = self.conv2(self.conv1(x))
        x = self.feature(x)
        result=OrderedDict()
        result_list.append(last_fm_list[-1])
        result[0]=result_list[0]
        result[1]=result_list[1]
        result[2]=result_list[2]
        return result

if __name__ == '__main__':
    net = MnasNet(width_mult=0.25)
    x = torch.randn(1,3,320,320)
    net(x)

