import argparse

parser = argparse.ArgumentParser(description="SAPNet_train_test")

# General Settings
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")  # default is 16
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30, 50, 80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/SAPNet/Model1", help='path to save models and log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')

# For test only
parser.add_argument("--test_data_path", type=str, default="datasets/test/Rain100H", help='path to testing data')
parser.add_argument("--output_path", type=str, default="results", help='path to save output images')

# For train only
parser.add_argument("--data_path", type=str, default="datasets/train/RainTrainH",
                    help='path to synthesized training data')
parser.add_argument("--data_path_real", type=str, default="D:/Code/AAAI_2022/datasets/real_input",
                    help='path to real rainy training data')
parser.add_argument("--use_contrast", type=bool, default=True,
                    help='use contrasive loss or not') # TODO: ablation on this
parser.add_argument("--use_stage1", type=bool, default=True,
                    help='use stage1: train on synthesized image') # TODO: ablation on this (more or less)
parser.add_argument("--use_stage2", type=bool, default=True,
                    help='use stage2: train on real image') # TODO: ablation on this (yes or no, more or less)
parser.add_argument("--use_ghost", type=bool, default=False,
                    help='use (multi-scale) ghost block or not')

# For model only
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')  # TODO: ablation on this
parser.add_argument("--num_of_SegClass", type=int, default=21,
                    help='Number of Segmentation Classes, default VOC = 21')

opt = parser.parse_args()
