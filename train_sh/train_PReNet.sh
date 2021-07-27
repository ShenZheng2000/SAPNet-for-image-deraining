#! bash

# Rain100H
python train_PReNet.py

# Rain100L
python train_PReNet.py --batch_size 32 --preprocess True --save_path logs/Rain100L/PReNet --data_path datasets/train/RainTrainL

# Rain12600
python train_PReNet.py --batch_size 32 --preprocess True --save_path logs/Rain1400/PReNet --data_path datasets/train/Rain12600
