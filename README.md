# SAPNet for image deraining

This repository contains the official Pytorch implementation of the paper:

"SAPNet: Segmentation-Aware Progressive Network for Perceptual Contrastive Image Deraining"

# Updates: 
- 2021.11.27: The pretrained weight has been uploaded.
- 2021.11.27: The video deraining result at challenging night-time conditions is at [Youtube](https://www.youtube.com/watch?v=tJSsICHpsfs)
- 2021.11.17: Arxiv Link is available at https://arxiv.org/abs/2111.08892
- 2021.11.10: This work is accepted and will be presented at [WACV 2022](https://wacv2022.thecvf.com/home).

# Abstract
Deep learning algorithms have recently achieved promising deraining performances on both the natural and synthetic rainy datasets. As an essential low-level pre-processing stage, a deraining network should clear the rain streaks and preserve the fine semantic details. However, most existing methods only consider low-level image restoration. That limits their performances at high-level tasks requiring precise semantic information. To address this issue, in this paper, we present a **segmentation-aware progressive network** (SAPNet) based upon contrastive learning for single image deraining. We start our method with a lightweight derain network formed with **progressive dilated units** (PDU). The PDU can significantly expand the receptive field and characterize multi-scale rain streaks without the heavy computation on multi-scale images. A fundamental aspect of this work is an **unsupervised background segmentation** (UBS) network initialized with ImageNet and Gaussian weights. The UBS can faithfully preserve an image's semantic information and improve the generalization ability to unseen photos. Furthermore, we introduce a **perceptual contrastive loss** (PCL) and a **learned perceptual image similarity loss** (LPISL) to regulate model learning. By exploiting the rainy image and groundtruth as the negative and the positive sample in the VGG-16 latent space, we bridge the fine semantic details between the derained image and the groundtruth in a fully constrained manner. Comprehensive experiments on synthetic and real-world rainy images show our model surpasses top-performing methods and aids object detection and semantic segmentation with considerable efficacy.

# Model Architecture
Click this following link to view the pdf.

![Model Architecture Figure](resources/main_figure.pdf)

# Comparison on Synthetic Rainy Images
From left to right, Rainy, PreNet (CVPR 2019), MSPFN (CVPR 2020), MPRNet (CVPR 2021), SAPNet (Ours)

<p float="left">
<p align="middle">
  <img src="resources/rain-064.jpg" width="170" />
  <img src="resources/PreNet-064.jpg" width="170" /> 
  <img src="resources/MSPFN-064.jpg" width="170" />
  <img src="resources/MPRNet-064.png" width="170" />
  <img src="resources/Ours-064.jpg" width="170" /> 
</p>

# Comparison on Real-World Rainy Images
From left to right, Rainy, PreNet (CVPR 2019), MSPFN (CVPR 2020), MPRNet (CVPR 2021), SAPNet (Ours)

<p float="left">
<p align="middle">
  <img src="resources/rainy_.jpg" width="170" />
  <img src="resources/PreNet.jpg" width="170" /> 
  <img src="resources/MSPFN.jpg" width="170" />
  <img src="resources/MPRNet.png" width="170" />
  <img src="resources/ours.jpg" width="170" /> 
</p>

# Prerequisites
* Windows or Linux
* CUDA 10.0, or CPU
* Python 3.6+
* Pytorch 1.2+
* torchvision 0.4+
* opencv-python
* scikit-image
* numpy

# Preparing Dataset
- Download training and testing dataset from either link 
[BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg#list/path=%2Fsharelink3792638399-290876125944720%2Fdatasets&parentPath=%2Fsharelink3792638399-290876125944720)
[OneDrive](https://onedrive.live.com/?cid=066ce859ab42dfa2&id=66CE859AB42DFA2%2130078&authkey=%21AIYIy8ZKL9kkmd4)

- Create a new folder called `dataset`.
- Create sub-folders called `train` and `test` under that folder. 
- Place the unzipped folders into `datasets/train/` (training data) and `datasets/test/` (testing data)

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

You can modify the content of `main.sh` according to your hyperparameters. For example.
```
python test.py \
      --save_path logs/SAPNet/Model11 \
      --test_data_path datasets/test/Rain100H \
      --output_path results/Rain100H/Model11 \
      --use_dilation True
```

# Hyperparameters
## General Hyperparameters
| Name       | Type  | Default             | Description |
|------------|-------|---------------------|-------------|
| preprocess | bool  | False               | Whether to preprocess the training images as h5 file            |
| batch_size | int   | 12                  | training batch size            |
| epochs     | int   | 100                 | training epochs            |
| milestone  | int   | [30,50,80]          | when to decay the learning rate            |
| lr         | float | 0.001               | learning rate            |
| save_path  | str   | logs/SAPNet/Model11 | where to save the weight            |
| save_freq  | int   | 1                   | frequency for saving the weight            |

## Train/Test Hypeparameters
| Name            | Type | Default                   | Description |
|-----------------|------|---------------------------|-------------|
| test_data_path  | str  | datasets/test/Rain100H    | testing data path           |
| output_path     | str  | results/Rain100H/Model11  | testing result path        |
| data_path       | str  | datasets/train/RainTrainH | training data path            |
| use_contrast    | bool | True                      | whether to use contrast learning            |
| use_seg_stage1  | bool | True                      | whether to use segmentation            |
| use_stage1      | bool | True                      | whether to use deraining            |
| use_dilation    | bool | True                      | whether to use dilation            |
| recurrent_iter  | int  | 6                         | numbers of recurrence            |
| num_of_SegClass | int  | 21                        | numbers of segmentation classes            |

# Citations
Please cite the following if you find this repository helpful. 
```
@inproceedings{zheng2022sapnet,
  title={SAPNet: Segmentation-Aware Progressive Network for Perceptual Contrastive Deraining},
  author={Zheng, Shen and Lu, Changjie and Wu, Yuxiong and Gupta, Gaurav},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={52--62},
  year={2022}
}
```

# Contact
Please reach zhengsh@kean.edu for further questions. You can also open an issue (prefered) or a pull request in this Github repository 

# Acknowledgement
This repository is borrowed heavily from [PreNet](https://github.com/csdwren/PReNet). Thanks for sharing!

# TODO List
- [x] List (important) hyperparameters
- [x] Upload Testing Dataset
- [x] Upload Arxiv Link
- [x] Upload BibTeX
- [x] Add hyperparameter explanation
- [x] Add Dependencies
- [x] Update Code with lpis loss
- [x] Upload Model Architecture Figure
- [x] Upload Visual Comparisons
- [x] Testing on Video
- [x] Upload Pretrained Weight 
- [ ] Finalize readme


