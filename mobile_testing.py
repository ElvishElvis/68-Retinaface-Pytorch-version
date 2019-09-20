from collections import OrderedDict
import torch.nn as nn
import torch
class mobileV1(nn.Module):
    def __init__(self):
        super(mobileV1, self).__init__()

        self.mmm = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=32*4, kernel_size=7, stride=4, padding=2, bias=False),
                    nn.BatchNorm2d(num_features=32*4, momentum=0.9),
                    nn.ReLU(inplace=True))


        self.mmm1 = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=4, padding=2, bias=False),
                    nn.BatchNorm2d(num_features=3, momentum=0.9),
                    nn.ReLU(inplace=True))
        self.mmm2 = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=32*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=32*4, momentum=0.9),
                    nn.ReLU(inplace=True))
            
        self.mobilenet0_conv0 = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=8*4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=8*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=8*4, out_channels=8*4, kernel_size=3, stride=1, padding=1, groups=8*4, bias=False),
                    nn.BatchNorm2d(num_features=8*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=8*4, out_channels=16*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=16*4, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv3 = nn.Sequential(
                    nn.Conv2d(in_channels=16*4, out_channels=16*4, kernel_size=3, stride=2, padding=1, groups=16*4, bias=False),
                    nn.BatchNorm2d(num_features=16*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv4 = nn.Sequential(
                    nn.Conv2d(in_channels=16*4, out_channels=32*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=32*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv5 = nn.Sequential(
                    nn.Conv2d(in_channels=32*4, out_channels=32*4, kernel_size=3, stride=1, padding=1, groups=32*4, bias=False),
                    nn.BatchNorm2d(num_features=32*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv6 = nn.Sequential(
                    nn.Conv2d(in_channels=32*4, out_channels=32*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=32*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv7 = nn.Sequential(
                    nn.Conv2d(in_channels=32*4, out_channels=32*4, kernel_size=3, stride=2, padding=1, groups=32*4, bias=False),
                    nn.BatchNorm2d(num_features=32*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv8 = nn.Sequential(
                    nn.Conv2d(in_channels=32*4, out_channels=64*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=64*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv9 = nn.Sequential(
                    nn.Conv2d(in_channels=64*4, out_channels=64*4, kernel_size=3, stride=1, padding=1, groups=64*4, bias=False),
                    nn.BatchNorm2d(num_features=64*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv10 = nn.Sequential(
                    nn.Conv2d(in_channels=64*4, out_channels=64*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=64*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv11 = nn.Sequential(
                    nn.Conv2d(in_channels=64*4, out_channels=64*4, kernel_size=3, stride=2, padding=1, groups=64*4, bias=False),
                    nn.BatchNorm2d(num_features=64*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv12 = nn.Sequential(
                    nn.Conv2d(in_channels=64*4, out_channels=128*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv13 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=128*4, kernel_size=3, stride=1, padding=1, groups=128*4, bias=False),
                    nn.BatchNorm2d(num_features=128*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv14 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=128*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv15 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=128*4, kernel_size=3, stride=1, padding=1, groups=128*4, bias=False),
                    nn.BatchNorm2d(num_features=128*4),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv16 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=128*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128*4, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv17 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=128*4, kernel_size=3, stride=1, padding=1, groups=128*4, bias=False),
                    nn.BatchNorm2d(num_features=128*4, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv18 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=128*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv19 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=128*4, kernel_size=3, stride=1, padding=1, groups=128*4, bias=False),
                    nn.BatchNorm2d(num_features=128*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv20 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=128*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv21 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=128*4, kernel_size=3, stride=1, padding=1, groups=128*4, bias=False),
                    nn.BatchNorm2d(num_features=128*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv22 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=128*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv23 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=128*4, kernel_size=3, stride=2, padding=1, groups=128*4, bias=False),
                    nn.BatchNorm2d(num_features=128*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv24 = nn.Sequential(
                    nn.Conv2d(in_channels=128*4, out_channels=256*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=256*4, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv25 = nn.Sequential(
                    nn.Conv2d(in_channels=256*4, out_channels=256*4, kernel_size=3, stride=1, padding=1, groups=256*4, bias=False),
                    nn.BatchNorm2d(num_features=256*4, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv26 = nn.Sequential(
                    nn.Conv2d(in_channels=256*4, out_channels=256*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=256*4, momentum=0.9),
                    nn.ReLU(inplace=True))
    def forward(self, x):
        result=OrderedDict()
        batchsize = x.shape[0]
        # k1=F.interpolate(k,(512,512),mode='nearest')
        # x = self.mobilenet0_conv0(x)
        # x = self.mobilenet0_conv1(x)
        # x = self.mobilenet0_conv2(x)
        
        # x = self.mobilenet0_conv3(x)
        # x = self.mobilenet0_conv4(x)
        # x=self.mmm1(x)
        x=self.mmm(x)
        # print(x.shape)
        x = self.mobilenet0_conv5(x)
        x = self.mobilenet0_conv6(x)
        x = self.mobilenet0_conv7(x)
        x = self.mobilenet0_conv8(x)
        x = self.mobilenet0_conv9(x)
        x10 = self.mobilenet0_conv10(x)
        x = self.mobilenet0_conv11(x10)
        x = self.mobilenet0_conv12(x)
        x = self.mobilenet0_conv13(x)
        x = self.mobilenet0_conv14(x)
        x = self.mobilenet0_conv15(x)
        x = self.mobilenet0_conv16(x)
        x = self.mobilenet0_conv17(x)
        x = self.mobilenet0_conv18(x)
        x = self.mobilenet0_conv19(x)
        x = self.mobilenet0_conv20(x)
        x = self.mobilenet0_conv21(x)
        x22 = self.mobilenet0_conv22(x)
        x = self.mobilenet0_conv23(x22)
        x = self.mobilenet0_conv24(x)
        x = self.mobilenet0_conv25(x)
        x26 = self.mobilenet0_conv26(x)
        result[1]=x10
        result[2]=x22
        result[3]=x26
        return result
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = mobileV1().to(device)
    
    #print(net)
    import time
    x = torch.randn(1,3,640,640).to(device)
    torch.cuda.synchronize()
    start=time.time()
    for i in range(10):
        net(x)
    torch.cuda.synchronize()
    print(time.time()-start)
    torch.save(net.state_dict(),'aaa.torch')


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import datetime

# def Conv_3x3(in_channels, out_channels, stride):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU6()
#     )

# def Conv_1x1(in_channels, out_channels, stride):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU6()
#     )

# def SepConv_3x3(in_channels, out_channels, stride):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
#         nn.BatchNorm2d(in_channels),
#         nn.ReLU6(),

#         nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(out_channels)
#     )

# class MBConv3_3x3(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):  
#         super(MBConv3_3x3, self).__init__()
#         mid_channels = int(3 * in_channels)

#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU6(),

#             nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU6(),

#             nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )

#         self.use_skip_connect = (1 == stride and in_channels == out_channels)

#     def forward(self, x):
#         if self.use_skip_connect:
#             return self.block(x) + x
#         else:
#             return self.block(x)    

# class MBConv3_5x5(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):  
#         super(MBConv3_5x5, self).__init__()
#         mid_channels = int(3 * in_channels)

#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU6(),

#             nn.Conv2d(mid_channels, mid_channels, 5, stride, 2, groups=mid_channels, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU6(),

#             nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )

#         self.use_skip_connect = (1 == stride and in_channels == out_channels)

#     def forward(self, x):
#         if self.use_skip_connect:
#             return self.block(x) + x
#         else:
#             return self.block(x) 

# class MBConv6_3x3(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):  
#         super(MBConv6_3x3, self).__init__()
#         mid_channels = int(6 * in_channels)

#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU6(),

#             nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU6(),

#             nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )

#         self.use_skip_connect = (1 == stride and in_channels == out_channels)

#     def forward(self, x):
#         if self.use_skip_connect:
#             return self.block(x) + x
#         else:
#             return self.block(x) 

# class MBConv6_5x5(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):  
#         super(MBConv6_5x5, self).__init__()
#         mid_channels = int(6 * in_channels)

#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU6(),

#             nn.Conv2d(mid_channels, mid_channels, 5, stride, 2, groups=mid_channels, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU6(),

#             nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )

#         self.use_skip_connect = (1 == stride and in_channels == out_channels)

#     def forward(self, x):
#         if self.use_skip_connect:
#             return self.block(x) + x
#         else:
#             return self.block(x) 

# class MnasNet(nn.Module):
#     def __init__(self,  width_mult=1.):
#         super(MnasNet, self).__init__()

#         self.out_channels = int(1280 * width_mult)

#         self.conv1 = Conv_3x3(3, int(32 * width_mult), 2)
#         self.conv2 = SepConv_3x3(int(32 * width_mult), int(16 * width_mult), 1)

#         self.feature1 = nn.Sequential(
#             self._make_layer(MBConv3_3x3, 3, int(16 * width_mult), int(24 * width_mult), 2),
#             self._make_layer(MBConv3_5x5, 3, int(24 * width_mult), int(64 * width_mult), 2)
            
#         )
#         self.feature2=nn.Sequential(
            
#             self._make_layer(MBConv6_5x5, 3, int(64 * width_mult), int(80 * width_mult), 2),
#             self._make_layer(MBConv6_3x3, 2, int(80 * width_mult), int(128 * width_mult), 1)
            
#             )
#         self.feature3=nn.Sequential(
            
#             self._make_layer(MBConv6_5x5, 4, int(128 * width_mult), int(192 * width_mult), 2),
#             self._make_layer(MBConv6_3x3, 1, int(192 * width_mult), int(256 * width_mult), 1))

#         self._initialize_weights()

#     def _make_layer(self, block, blocks, in_channels, out_channels, stride=1):
#         strides = [stride] + [1] * (blocks - 1)
#         layers = []
#         for _stride in strides:
#             layers.append(block(in_channels, out_channels, _stride))
#             in_channels = out_channels
        
#         return nn.Sequential(*layers)

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_() 

#     def forward(self, x):
#         x = self.conv2(self.conv1(x))
#         # print(x.shape)
#         x = self.feature1(x)
#         # print(x.shape)
#         x = self.feature2(x)
#         # print(x.shape)
#         x = self.feature3(x)
#         # print(x.shape)

#         return x

# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net = MnasNet().to(device)
#     #print(net)
#     import time
#     torch.cuda.synchronize()
#     x = torch.randn(1,3,640,640).to(device)
#     start=time.time()
#     for i in range(10):
#         net(x)
#     torch.cuda.synchronize()
#     print(time.time()-start)
    
#     torch.save(net.state_dict(),'aaa.torch')
#     # print(net)
#     #print(y)

