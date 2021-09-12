from .resnet import resnet101
import torch.nn as nn
import torch.nn.functional as F
import torch

class fpn_module(nn.Module):
    def __init__(self, numClass):
        super(fpn_module, self).__init__()
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Classify layers
        self.classify = nn.Conv2d(128*4, numClass, kernel_size=3, stride=1, padding=1)


    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), mode='bilinear', align_corners=True)
        p4 = F.interpolate(p4, size=(H, W), mode='bilinear', align_corners=True)
        p3 = F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=True)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, c2, c3, c4, c5):
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p5 = self.smooth1_2(self.smooth1_1(p5))
        p4 = self.smooth2_2(self.smooth2_1(p4))
        p3 = self.smooth3_2(self.smooth3_1(p3))
        p2 = self.smooth4_2(self.smooth4_1(p2))

        # Classify
        output = self.classify(self._concatenate(p5, p4, p3, p2))

        return output


class fpn(nn.Module):
    def __init__(self, numClass):
        super(fpn, self).__init__()
        # Resnet
        self.resnet = resnet101(True)

        # fpn module
        self.fpn = fpn_module(numClass)

        # init fpn
        for m in self.fpn.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Top-down
        c2, c3, c4, c5 = self.resnet.forward(x)
        return self.fpn.forward(c2, c3, c4, c5)






