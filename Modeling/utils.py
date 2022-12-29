import math
import torch
import re
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
# from skimage.metrics import peak_signal_noise_ratio # NOTE: use this for the latest sklearn version
import os
import glob
import torch.nn.functional as F
import cv2

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        # PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range) # NOTE: use this for the latest sklearn version
    return (PSNR/Img.shape[0])


def normalize(data):
    return data / 255.


def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return  False


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class FocalLoss(nn.Module):

    # def __init__(self, device, gamma=0, eps=1e-7, size_average=True):
    def __init__(self, gamma=0, eps=1e-7, size_average=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.reduce = reduce
        # self.device = device

    def forward(self, input, target):
        # y = one_hot(target, input.size(1), self.device)
        y = one_hot(target, input.size(1))
        probs = F.softmax(input, dim=1)
        probs = (probs * y).sum(1)  # dimension ???
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.reduce:
            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
        else:
            loss = batch_loss
        return loss

def one_hot(index, classes):
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]

    # mask = torch.Tensor(size).fill_(0).to(device)
    if torch.cuda.is_available():
        mask = torch.Tensor(size).fill_(0).cuda()
    else:
        mask = torch.Tensor(size).fill_(0)
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)


def get_NoGT_target(inputs):
    sfmx_inputs = F.log_softmax(inputs, dim=1)
    target = torch.argmax(sfmx_inputs, dim=1)
    return target

def rgb_demean(inputs):
    rgb_mean = np.array([0.48109378172, 0.4575245789, 0.4078705409]).reshape((3, 1, 1))
    inputs = inputs - rgb_mean  # inputs in [0,1]
    return inputs

def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_CUBIC)
    return new_target


# Define the laplacian matrix (remember to put float tensor into cuda!)

def Laplacian(x):
    '''
    weight = tf.constant([
        [[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
         [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]],
        [[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
         [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]],
        [[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
         [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]]
    ])

    frame = tf.nn.conv2d(x, weight, [1, 1, 1, 1], padding='SAME')
    '''
    weight = torch.FloatTensor([
        [[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
         [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]],
        [[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
         [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]],
        [[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
         [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]]
    ])

    if torch.cuda.is_available():
        weight = weight.cuda()

    frame = F.conv2d(x, weight, stride=1, padding=0)
    return frame


# Define the modified mse loss
def inference_mse_loss(frame_hr, frame_sr):
    #content_base_loss = tf.reduce_mean(tf.sqrt((frame_hr - frame_sr) ** 2 + (1e-3) ** 2))
    content_base_loss = torch.mean(torch.sqrt((frame_hr - frame_sr) ** 2 + (1e-3) ** 2))
    return torch.mean(content_base_loss)

def laplacian_loss(target, out):
    # Edge for clean image
    target_train_edge = Laplacian(target)

    # Edge for derained image
    out_train_edge = Laplacian(out)

    # Edge loss
    edge_loss = inference_mse_loss(target_train_edge, out_train_edge)

    # return
    return edge_loss
