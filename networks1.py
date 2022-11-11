import torch.nn as nn
import torch.nn.functional as F
import torch
class BasicConv2d1(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)
class inception_module1(nn.Module):
    # referred from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
    def __init__(self, in_channels, pool_features):
        super(inception_module1, self).__init__()
        self.branch1x1 = BasicConv2d1(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d1(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d1(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d1(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d1(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d1(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d1(in_channels, pool_features, kernel_size=1)
        self.conv2d = nn.Conv2d(225, 1, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs = torch.cat(outputs, 1)
        outputs = self.conv2d(outputs)
        return outputs
class DownNet1(nn.Module):
    def __init__(self):
        super(DownNet1, self).__init__()
        layers = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU()]
        for i in range(7):
            layers.append(nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, 3, 1, 1))
        layers.append(nn.Conv2d(64, 64, 3, 1, 1))
        layers.append(nn.ReLU())
        for i in range(6):
            layers.append(nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, 3, 1, 1))
        layers.append(nn.Conv2d(64, 64, 3, 1, 1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 1, 3, 1, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        out = self.net(x)
        # print('out=', out.shape)
        return out

class BRDNet1(nn.Module):
    def __init__(self):
        super(BRDNet1, self).__init__()
        # self.upnet = UpNet1()
        self.inception_module1 = inception_module1(in_channels=1, pool_features=1)
        self.dwnet = DownNet1()
        self.conv = nn.Conv2d(2, 1, 3, 1, 1)
    def forward(self, x):
        out1 = self.inception_module1(x)
        out2 = self.dwnet(x)
        out1 = x - out1
        out2 = x - out2
        out = torch.cat((out1, out2), 1)
        out = self.conv(out)
        out = x - out
        return out


class BRDNet2(nn.Module):
    def __init__(self):
        super(BRDNet2, self).__init__()
        # self.upnet = UpNet1()
        self.inception_module2 = inception_module1(in_channels=1, pool_features=1)
        self.dwnet = DownNet1()
        self.conv = nn.Conv2d(2, 1, 3, 1, 1)
    def forward(self, x):
        # out1 = self.upnet(x)
        out1 = self.inception_module2(x)
        out2 = self.dwnet(x)
        out1 = x - out1
        out2 = x - out2
        out = torch.cat((out1, out2), 1)
        out = self.conv(out)
        out = x - out
        return out


class BRDNet3(nn.Module):
    def __init__(self):
        super(BRDNet3, self).__init__()
        # self.upnet = UpNet1()
        self.inception_module3 = inception_module1(in_channels=1, pool_features=1)
        self.dwnet = DownNet1()
        self.conv = nn.Conv2d(2, 1, 3, 1, 1)
    def forward(self, x):
        # out1 = self.upnet(x)
        out1 = self.inception_module3(x)
        out2 = self.dwnet(x)
        out1 = x - out1
        out2 = x - out2
        out = torch.cat((out1, out2), 1)
        out = self.conv(out)
        out = x - out
        return out

class BRDNet4(nn.Module):
    def __init__(self):
        super(BRDNet4, self).__init__()
        # self.upnet = UpNet1()
        self.inception_module4 = inception_module1(in_channels=1, pool_features=1)
        self.dwnet = DownNet1()
        self.conv = nn.Conv2d(2, 1, 3, 1, 1)
    def forward(self, x):
        # out1 = self.upnet(x)
        out1 = self.inception_module4(x)
        out2 = self.dwnet(x)
        out1 = x - out1
        out2 = x - out2
        out = torch.cat((out1, out2), 1)
        out = self.conv(out)
        out = x - out
        return out

class BRDNet5(nn.Module):
    def __init__(self):
        super(BRDNet5, self).__init__()
        # self.upnet = UpNet1()
        self.inception_module4 = inception_module1(in_channels=1, pool_features=1)
        self.dwnet = DownNet1()
        self.conv = nn.Conv2d(2, 1, 3, 1, 1)
    def forward(self, x):
        # out1 = self.upnet(x)
        out1 = self.inception_module4(x)
        out2 = self.dwnet(x)
        out1 = x - out1
        out2 = x - out2
        out = torch.cat((out1, out2), 1)
        out = self.conv(out)
        out = x - out
        return out


class RED_CNN(nn.Module):
    def __init__(self, out_ch=64):
        super(RED_CNN, self).__init__()
        self.BRDNet1 = BRDNet1()
        self.BRDNet2 = BRDNet2()
        self.BRDNet3 = BRDNet3()
        self.BRDNet4 = BRDNet4()
        self.BRDNet5 = BRDNet5()
        self.conv1 = nn.Conv2d(2, 1, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 1, 3, 1, 1)
        self.conv3 = nn.Conv2d(4, 1, 3, 1, 1)
        self.conv4 = nn.Conv2d(5, 1, 3, 1, 1)
        self.conv5 = nn.Conv2d(6, 1, 3, 1, 1)

    def forward(self, x):
        out = []
        out_ = []
        block_1 = x.clone()
        out1 = self.BRDNet1(x)
        # out1 += block_1
        out1 = torch.cat((block_1, out1), 1)
        out1 = self.conv1(out1)
        out.append(out1)
        out1_ = x - 0.4 * out1
        out_.append(out1_)

        # out2 = self.BRDNet2(out1)
        # # out2 += block_1
        # out2 = torch.cat((block_1, out2, out1), 1)
        out2 = self.BRDNet2(out1_)
        out2 = torch.cat((block_1, out2, out1_), 1)

        out2 = self.conv2(out2)
        out.append(out2)
        out2_ = out1_ - 0.2 * out2
        out_.append(out2_)

        # out3 = self.BRDNet3(out2)
        # # out3 += block_1
        # out3 = torch.cat((block_1, out3, out2, out1), 1)
        out3 = self.BRDNet3(out2_)
        out3 = torch.cat((block_1, out3, out2_, out1_), 1)

        out3 = self.conv3(out3)
        out.append(out3)
        out3_ = out2_ - 0.2 * out3
        out_.append(out3_)

        # out4 = self.BRDNet4(out3)
        # # out3 += block_1
        # out4 = torch.cat((block_1, out4, out3, out2, out1), 1)
        out4 = self.BRDNet4(out3_)
        out4 = torch.cat((block_1, out4, out3_, out2_, out1_), 1)

        out4 = self.conv4(out4)
        out.append(out4)
        out4_ = out3_ - 0.1 * out4
        out_.append(out4_)

        # out5 = self.BRDNet5(out4)
        # # out3 += block_1
        # out5 = torch.cat((block_1, out5, out4, out3, out2, out1), 1)
        out5 = self.BRDNet5(out4)
        out5 = torch.cat((block_1, out5, out4_, out3_, out2_, out1_), 1)

        out5 = self.conv5(out5)
        out.append(out5)
        out5_ = out4_ - 0.1 * out5
        out_.append(out5_)
        return out, out_