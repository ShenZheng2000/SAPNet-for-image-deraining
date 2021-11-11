# SAPNet

This repository contains the official Pytorch implementation of the paper:
"SAPNet: Segmentation-Aware Progressive Network for Contrastive Image Deraining"

# Abstract
Deep learning algorithms have recently achieved promising deraining performances on both the natural and synthetic rainy datasets. As an essential low-level pre-processing stage, a deraining network should clear the rain streaks and preserve the fine semantic details. However, most existing methods only consider low-level image restoration. That limits their performances at high-level tasks requiring precise semantic information. To address this issue, in this paper, we present a segmentation-aware progressive network (SAPNet) based upon contrastive learning for single image deraining. We start our method with a lightweight derain network formed with progressive dilated units (PDU). The PDU can significantly expand the receptive field and characterize multi-scale rain streaks without the heavy computation on multi-scale images. A fundamental aspect of this work is an unsupervised background segmentation (UBS) network initialized with ImageNet and Gaussian weights. The UBS can faithfully preserve an image's semantic information and improve the generalization ability to unseen photos. Furthermore, we introduce a perceptual contrastive loss (PCL) and a learned perceptual image similarity loss (LPISL) to regulate model learning. By exploiting the rainy image and groundtruth as the negative and the positive sample in the VGG-16 latent space, we bridge the fine semantic details between the derained image and the groundtruth in a fully constrained manner. Comprehensive experiments on synthetic and real-world rainy images show our model surpasses top-performing methods and aids object detection and semantic segmentation with considerable efficacy.



# Preparing Dataset
First, download training and testing dataset from either link 
[BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg#list/path=%2Fsharelink3792638399-290876125944720%2Fdatasets&parentPath=%2Fsharelink3792638399-290876125944720)
[OneDrive](https://onedrive.live.com/?cid=066ce859ab42dfa2&id=66CE859AB42DFA2%2130078&authkey=%21AIYIy8ZKL9kkmd4)

Next, create new folders called dataset. Then create sub-folders called train and test under that folder. Finally, place the unzipped folders into `./datasets/train/` (training data) and `./datasets/test/` (testing data)

# Training
Run the following script in terminal
```
python train.py
```

# Testing
Run the following script in terminal
```
bash main.sh
```

# Hyperparameters
## General Hyperparameters
| Name       | Type  | Default             | Description |
|------------|-------|---------------------|-------------|
| preprocess | bool  | False               |             |
| batch_size | int   | 12                  |             |
| epochs     | int   | 100                 |             |
| milestone  | int   | [30,50,80]          |             |
| lr         | float | 0.001               |             |
| save_path  | str   | logs/SAPNet/Model11 |             |
| save_freq  | int   | 1                   |             |

## Train/Test Hypeparameters
| Name            | Type | Default                   | Description |
|-----------------|------|---------------------------|-------------|
| test_data_path  | str  | datasets/test/Rain100H    |             |
| output_path     | str  | results/Rain100H/Model11  |             |
| data_path       | str  | datasets/train/RainTrainH |             |
| use_contrast    | bool | True                      |             |
| use_seg_stage1  | bool | True                      |             |
| use_stage1      | bool | True                      |             |
| use_dilation    | bool | True                      |             |
| recurrent_iter  | int  | 6                         |             |
| num_of_SegClass | int  | 21                        |             |

# Contact
Please reach zhengsh@kean.edu for further questions. You can also open an issue (prefered) or a pull request in this Github repository 

# Acknowledgement
This repository is borrowed heavily from [PreNet](https://github.com/csdwren/PReNet). Thanks for sharing!

# TODO List
- [x] Upload Pretrained Weight 
- [ ] Add Visual Comparisons
- [ ] Add References
- [ ] Upload Arxiv Link
- [ ] Upload BibTeX
