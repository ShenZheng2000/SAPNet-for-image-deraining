# Purpose: calculate psnr and ssim of two different datasets
# keep this file in mind D:/Code/SOTA/Ref_Data/Rain100H/

import os
import numpy as np
import math
import cv2

import time

def oldblock(code):
    return "%03d" % code

# 当中是你的程序
def psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    return 10 * math.log10(255.0 * 255.0 / mse)

'''
def ssim(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 7)
    c2 = np.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom
'''

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

path1 = 'D:/Code/SOTA/Ref_Data/PreNet8/'  # 指定输出结果文件夹
path2 = 'D:/Code/SOTA/Ref_Data/Rain100H/'  # 指定原图文件夹

f_nums = len(os.listdir(path1))
list_psnr = []
list_ssim = []

# Consider Use 1, 2, 3 as key for searching 001, 002, 003
for i in range(1, f_nums + 1):
    img_a = cv2.imread(path1 +  'rain-' + oldblock(i) + '.png')
    img_b = cv2.imread(path2 + 'norain-' +  oldblock(i) + '.png')
    psnr_num = psnr(img_a, img_b)
    ssim_num = calculate_ssim(img_a, img_b)
    list_ssim.append(ssim_num)
    list_psnr.append(psnr_num)

print("平均PSNR:", np.mean(list_psnr))  # ,list_psnr)
print("平均SSIM:", np.mean(list_ssim))  # ,list_ssim)


