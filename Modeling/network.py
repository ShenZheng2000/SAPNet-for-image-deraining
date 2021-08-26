# Purpose: Use channel attention layers with skip connections

# PyTorch lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Depthwise Separable Convolution
class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

# multi-scale ghost module
class MultiGhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(MultiGhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, init_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


# Ghost Convolution
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

# Pixel attention
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

# Channel Attention
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# Channel Residual Attention Layer
class CRALayer(nn.Module):
    def __init__(self, channel, reduction): # channel = 32, reduction = 16
        super(CRALayer, self).__init__()
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Feature Channel Rescale
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
        )
        # 1 X 1 Convolution inside Skip Connection
        self.conv_1_1 = nn.Conv2d(channel, channel, 1, padding=0, bias=False)
        # Sigmoid Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        res = self.conv_1_1(y)
        y = self.conv_du(y)
        y += res
        y = self.sigmoid(y)
        return x * y

# Pixel Residual Attention Layer
class PRALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(PRALayer, self).__init__()
        # Feature Channel Rescale
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
        )
        # 1 X 1 Convolution inside Skip Connection
        self.conv_1_1 = nn.Conv2d(channel, channel, 1, padding=0, bias=False)
        # Sigmoid Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.conv_1_1(x)
        y = self.conv_du(x)
        y += res
        y = self.sigmoid(y)
        return x * y



class SAPNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_dilation = False):
        super(SAPNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_dilation = use_dilation

        if self.use_dilation:
            dilation = [1, 2, 4, 8, 16]
            padding = dilation
        else:
            dilation = [1, 1, 1, 1, 1]
            padding = dilation

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        # Residual Attention block
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding=padding[0], dilation=dilation[0]),
            CRALayer(channel=32, reduction=16),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=padding[0], dilation=dilation[0]),
            CRALayer(channel=32, reduction=16),
            nn.ReLU()
        )

        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding=padding[1], dilation=dilation[1]),
            CRALayer(channel=32, reduction=16),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=padding[1], dilation=dilation[1]),
            CRALayer(channel=32, reduction=16),
            nn.ReLU()
        )

        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding=padding[2], dilation=dilation[2]),
            CRALayer(channel=32, reduction=16),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=padding[2], dilation=dilation[2]),
            CRALayer(channel=32, reduction=16),
            nn.ReLU()
        )

        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding=padding[3], dilation=dilation[3]),
            CRALayer(channel=32, reduction=16),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=padding[3], dilation=dilation[3]),
            CRALayer(channel=32, reduction=16),
            nn.ReLU()
        )

        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding=padding[4], dilation=dilation[4]),
            CRALayer(channel=32, reduction=16),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=padding[4], dilation=dilation[4]),
            CRALayer(channel=32, reduction=16),
            nn.ReLU()
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        h = h.to(device)
        c = c.to(device)

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            '''if self.use_Contrast:
                x = self.mix(fea1 = input, fea2 = x)''' # input: rain image, x: derained image
            x = x + input

            x_list.append(x)

        return x, x_list




'''
class SAPNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_Contrast=False, use_ghost = False, use_multi = False):
        super(SAPNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_Contrast = use_Contrast
        self.use_ghost = use_ghost
        self.use_multi = use_multi
        if self.use_ghost:
            if self.use_multi:
                self.block = MultiGhostModule
            else:
                self.block = GhostModule
        else:
            self.block = CSDN_Tem

        self.conv0 = nn.Sequential(
            #nn.Conv2d(6, 32, 3, 1, 1),
            self.block(6, 32),
            nn.ReLU()
        )

        # Residual Attention block
        self.res_conv = nn.Sequential(
            #nn.Conv2d(32, 32, 3, 1, 1),
            self.block(32, 32),
            nn.ReLU(),
            #nn.Conv2d(32, 32, 3, 1, 1),
            self.block(32, 32),
            #CRALayer(channel=32, reduction=16),
            #PRALayer(channel=32, reduction=16),
            nn.ReLU()
        )

        self.conv_i = nn.Sequential(
            #nn.Conv2d(32 + 32, 32, 3, 1, 1),
            self.block(32 + 32, 32),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            #nn.Conv2d(32 + 32, 32, 3, 1, 1),
            self.block(32 + 32, 32),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            #nn.Conv2d(32 + 32, 32, 3, 1, 1),
            self.block(32 + 32, 32),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            #nn.Conv2d(32 + 32, 32, 3, 1, 1),
            self.block(32 + 32, 32),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            #nn.Conv2d(32, 3, 3, 1, 1),
            self.block(32, 3),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        h = h.to(device)
        c = c.to(device)

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv(x) + resx)
            resx = x
            x = F.relu(self.res_conv(x) + resx)
            resx = x
            x = F.relu(self.res_conv(x) + resx)
            resx = x
            x = F.relu(self.res_conv(x) + resx)
            resx = x
            x = F.relu(self.res_conv(x) + resx)
            x = self.conv(x)

            if self.use_Contrast:
                x = self.mix(fea1 = input, fea2 = x) # input: rain image, x: derained image
            else:
                x = x + input

            x_list.append(x)

        return x, x_list


'''
