#!/bin/bash

# SAPNet 1
python test.py  --logdir logs/SAPNet/Model1 --save_path results/SAPNet/Model1 --data_path datasets/test/Rain100H/rainy

# SAPNet 2
python test.py  --logdir logs/SAPNet/Model2 --save_path results/SAPNet/Model2 --data_path datasets/test/Rain100H/rainy

# SAPNet 3
python test.py  --logdir logs/SAPNet/Model3 --save_path results/SAPNet/Model3 --data_path datasets/test/Rain100H/rainy


python test.py  --logdir logs/SAPNet/Model1 --save_path results/SAPNet/Model1 --data_path datasets/real_input/train
