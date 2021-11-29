import torch
from torch.nn import L1Loss
from torch.nn.modules.loss import _Loss
import torch.nn as nn
from torchvision import models

# context.set_context(mode=context.GRAPH_MODE,
#                     device_target="Ascend",
#                     device_id=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ContrastLoss(_Loss):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.l1 = L1Loss().to(device)

    def forward(self, pred, pos, neg):
        loss = 0

        d_ap = self.l1(pred, pos) 
        d_an = self.l1(pred, neg) 
        contrastive = d_ap / (d_an + 1e-7)

        loss += contrastive

        return loss

if __name__ == '__main__':
    t1 = torch.ones([1,3,64,64])
    t2 = torch.zeros([1,3,64,64])
    t3 = torch.zeros([1,3,64,64])

    LC = ContrastLoss()

    print(LC(t1,t2,t3))
