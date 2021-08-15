#!/bin/bash

# First test if contrastive regularization is helpful

# Model1
python train.py --use_contrast False --save_path logs/SAPNet/Model1

# Model2
python train.py --use_contrast True --save_path logs/SAPNet/Model2

# Model 3
python train.py --use_contrast True --use_stage2 False --save_path logs/SAPNet/Model3

