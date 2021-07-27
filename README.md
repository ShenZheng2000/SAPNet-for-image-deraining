# SAPNet

This repository contains the official Pytorch implementation of the paper:
"SAPNet: Segmentation-Aware Progressive Network for Single Image Deraining"

# Abstract
Removing rains from an image has been an open yet challenging computer vision problem owing to its ill-posed nature. In recent years, deep learning algorithms have achieved notable deraining performances on both the real and synthetic rainy dataset. However, most methods only consider low-level image restoration and fail to accommo date real-world demands. Instead, task-driven approaches require paired annotations which are expensive to obtain. In this paper, we propose a segmentation-aware progres sive network (SAPNet) for single image deraining. Firstly, we introduce a recurrent progressive network with a novel channel residual attention (CRA) block. That design al lows us to remove rain streaks progressively with reduced parameters. Secondly, we design an unsupervised rain streak segmentation network (URSS) to outline and sep arate rain streaks. To further restore the edge informa tion of a rain image, we adopt canny edge loss in the cost function. Extensive experiments on the synthetic and real rain images show our model qualitatively and quantitatively outperforms other top-performing models. Notably, our model achieves the best PSNR and SSIM and becomes the new state-of-the-arts for Rain100L, Rain100H, and Rain12 benchmark dataset. We also visually demonstrate that our model aids detection and segmentation with considerable efficacy. A Pytorch implementation will be publicly avail able.


# Preparing Dataset
First, download training and testing dataset from either link 
[BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg#list/path=%2Fsharelink3792638399-290876125944720%2Fdatasets&parentPath=%2Fsharelink3792638399-290876125944720)
[OneDrive](https://onedrive.live.com/?cid=066ce859ab42dfa2&id=66CE859AB42DFA2%2130078&authkey=%21AIYIy8ZKL9kkmd4)

Next, create new folders called dataset. Then create sub-folders called train and test under that folder. Finally, place the unzipped folders into `./datasets/train/` (training data) and `./datasets/test/` (testing data)

# Training
Run the following script in terminal
`python train.py`

# Testing
TO DO

# Citation
TO DO
