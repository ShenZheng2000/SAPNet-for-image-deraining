import torch
import torch.nn as nn

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()


    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + \
              fea2 * (1 - mix_factor.expand_as(fea2))
        return out


if __name__ == "__main__":

    tensor_one = torch.ones(1, 256, 64, 64) # haze images
    tensor_two = torch.zeros(1, 256, 64, 64) # dehazed images
    mix1 = Mix(m=-0.8)



