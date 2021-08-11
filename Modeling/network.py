# Purpose: Use channel attention layers with skip connections

# PyTorch lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Mix import Mix


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, recurrent_iter=6, use_Contrast=False):
        super(SAPNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_Contrast = use_Contrast
        self.mix = Mix(m=-0.8)

        self.conv0 = nn.Sequential(
            #nn.Conv2d(6, 32, 3, 1, 1),
            CSDN_Tem(6, 32),
            nn.ReLU()
        )

        # Residual Attention block
        self.res_conv1 = nn.Sequential(
            #nn.Conv2d(32, 32, 3, 1, 1),
            CSDN_Tem(32, 32),
            CRALayer(channel=32, reduction=16),
            PRALayer(channel=32, reduction=16),
            nn.ReLU(),
            #nn.Conv2d(32, 32, 3, 1, 1),
            CSDN_Tem(32, 32),
            CRALayer(channel=32, reduction=16),
            PRALayer(channel=32, reduction=16),
            nn.ReLU()
        )

        self.res_conv2 = nn.Sequential(
            #nn.Conv2d(32, 32, 3, 1, 1),
            CSDN_Tem(32, 32),
            CRALayer(channel=32, reduction=16),
            PRALayer(channel=32, reduction=16),
            nn.ReLU(),
            #nn.Conv2d(32, 32, 3, 1, 1),
            CSDN_Tem(32, 32),
            CRALayer(channel=32, reduction=16),
            PRALayer(channel=32, reduction=16),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            #nn.Conv2d(32, 32, 3, 1, 1),
            CSDN_Tem(32, 32),
            CRALayer(channel=32, reduction=16),
            PRALayer(channel=32, reduction=16),
            nn.ReLU(),
            #nn.Conv2d(32, 32, 3, 1, 1),
            CSDN_Tem(32, 32),
            CRALayer(channel=32, reduction=16),
            PRALayer(channel=32, reduction=16),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            #nn.Conv2d(32, 32, 3, 1, 1),
            CSDN_Tem(32, 32),
            CRALayer(channel=32, reduction=16),
            PRALayer(channel=32, reduction=16),
            nn.ReLU(),
            #nn.Conv2d(32, 32, 3, 1, 1),
            CSDN_Tem(32, 32),
            CRALayer(channel=32, reduction=16),
            PRALayer(channel=32, reduction=16),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            #nn.Conv2d(32, 32, 3, 1, 1),
            CSDN_Tem(32, 32),
            CRALayer(channel=32, reduction=16),
            PRALayer(channel=32, reduction=16),
            nn.ReLU(),
            #nn.Conv2d(32, 32, 3, 1, 1),
            CSDN_Tem(32, 32),
            CRALayer(channel=32, reduction=16),
            PRALayer(channel=32, reduction=16),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            #nn.Conv2d(32 + 32, 32, 3, 1, 1),
            CSDN_Tem(32 + 32, 32),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            #nn.Conv2d(32 + 32, 32, 3, 1, 1),
            CSDN_Tem(32 + 32, 32),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            #nn.Conv2d(32 + 32, 32, 3, 1, 1),
            CSDN_Tem(32 + 32, 32),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            #nn.Conv2d(32 + 32, 32, 3, 1, 1),
            CSDN_Tem(32 + 32, 32),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            #nn.Conv2d(32, 3, 3, 1, 1),
            CSDN_Tem(32, 3),
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

            if self.use_Contrast:
                x = self.mix(fea1 = input, fea2 = x) # input: rain image, x: derained image
            else:
                x = x + input

            x_list.append(x)

        return x, x_list



