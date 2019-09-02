from collections import OrderedDict
import torch.nn as nn
import torch
class mobileV1(nn.Module):
    def __init__(self):
        super(mobileV1, self).__init__()
            
        self.mobilenet0_conv0 = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=8, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, groups=8, bias=False),
                    nn.BatchNorm2d(num_features=8, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=16, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv3 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, groups=16, bias=False),
                    nn.BatchNorm2d(num_features=16, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv4 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=32, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv5 = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
                    nn.BatchNorm2d(num_features=32, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv6 = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=32, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv7 = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
                    nn.BatchNorm2d(num_features=32, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv8 = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=64, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv9 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
                    nn.BatchNorm2d(num_features=64, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv10 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=64, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv11 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64, bias=False),
                    nn.BatchNorm2d(num_features=64, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv12 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv13 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv14 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv15 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv16 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv17 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv18 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv19 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv20 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv21 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv22 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv23 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv24 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=256, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv25 = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256, bias=False),
                    nn.BatchNorm2d(num_features=256, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv26 = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=256, momentum=0.9),
                    nn.ReLU(inplace=True))
    def forward(self, x):
        result_=OrderedDict()
        batchsize = x.shape[0]
        # k1=F.interpolate(k,(512,512),mode='nearest')
        x = self.mobilenet0_conv0(x)
        x = self.mobilenet0_conv1(x)
        x = self.mobilenet0_conv2(x)
        x = self.mobilenet0_conv3(x)
        x = self.mobilenet0_conv4(x)
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
        result_[1]=x10
        result_[2]=x22
        result_[3]=x26
        return result_
if __name__ == "__main__":
    from thop import profile
    net = mobileV1()
    from thop import profile
    
    from thop import clever_format
    # x = torch.randn(1,3,320,320)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(net, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(params)
    print(flops)
