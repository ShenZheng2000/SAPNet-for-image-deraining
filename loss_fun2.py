import torch
from torch.nn import L1Loss
from torch.nn.modules.loss import _Loss
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features # TODO: decide pretrained true or false
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out

class ContrastLoss(_Loss):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg16().to(device)
        self.l1 = L1Loss().to(device)
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, pred, pos, neg):
        pred_vgg, pos_vgg, neg_vgg = self.vgg(pred), self.vgg(pos), self.vgg(neg)
        loss = 0

        for i in range(len(pred_vgg)):
            d_ap = self.l1(pred_vgg[i], pos_vgg[i])
            d_an = self.l1(pred_vgg[i], neg_vgg[i])
            contrastive = d_ap / (d_an + 1e-7)

            loss += self.weights[i] * contrastive
        return loss

if __name__ == '__main__':
    t1 = torch.ones([1,3,64,64])
    t2 = torch.zeros([1,3,64,64])
    t3 = torch.ones([1,3,64,64])

    LC = ContrastLoss()

    print(LC(t1,t2,t3))
