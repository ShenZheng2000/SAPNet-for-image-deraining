# SAPNet

This repository contains the official Pytorch implementation of the paper:
"SAPNet: Segmentation-Aware Progressive Network for Contrastive Image Deraining"

# Abstract
Removing rains from an image has been an open yet challenging computer vision task. Recently, deep learning algorithms have achieved promising deraining performances on both the natural and synthetic rainy datasets. However, most existing methods only consider low-level image restoration, limiting their high-level detection and segmentation applications. Instead, task-driven approaches require heavy amounts of paired annotations that are expensive to obtain and hard to generalize. Furthermore, previous image deraining methods only use groundtruth images for supervised training and neglect the background details in the original rainy image.

This paper presents a segmentation-aware progressive network (SAPNet) based upon contrastive learning for single image deraining. We first introduce a recurrent progressive network with a new channel residual attention (CRA) block and dilated convolution. This architecture allows us to characterize multi-scale rain streaks progressively with affordable model size. Secondly, we design an unsupervised background segmentation network (UBS) to preserve the semantic information of an image during intensive rain removal. Finally, we introduce a novel perceptual contrastive loss (PCL) to ensure that the derained image is pulled to the groundtruth and is pushed from the rainy images in the VGG-16 latent space. Comprehensive experiments on synthetic and real-world rainy datasets show our model surpasses top-performing methods qualitatively and quantitatively. We also demonstrate that our model aids object detection and semantic segmentation with considerable efficacy. 


# Preparing Dataset
First, download training and testing dataset from either link 
[BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg#list/path=%2Fsharelink3792638399-290876125944720%2Fdatasets&parentPath=%2Fsharelink3792638399-290876125944720)
[OneDrive](https://onedrive.live.com/?cid=066ce859ab42dfa2&id=66CE859AB42DFA2%2130078&authkey=%21AIYIy8ZKL9kkmd4)

Next, create new folders called dataset. Then create sub-folders called train and test under that folder. Finally, place the unzipped folders into `./datasets/train/` (training data) and `./datasets/test/` (testing data)

# Training
Run the following script in terminal
`python train.py`

# Testing
Run the following script in terminal
`bash main.sh`

# Citation
TO DO
