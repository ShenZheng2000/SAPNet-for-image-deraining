# This is Model1 of SAPNet.
# The derain net uses depthwise conv
# The seg net uses fpn

import argparse
import torch.optim as optim
import torchvision.utils as utils
from torch.utils.data import DataLoader
from Modeling.DerainDataset import *
from Modeling.utils import *
from Modeling.Contrastive.Mix import *
from torch.optim.lr_scheduler import MultiStepLR
from Modeling.SSIM import SSIM
from Modeling.network import *
from Modeling.fpn import *
import sys
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description="SAPNet_train")
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size") # default is 16
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30, 50, 80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/SAPNet/Model1", help='path to save models and log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--data_path", type=str, default="datasets/train/RainTrainH", help='path to training data')
parser.add_argument("--data_path_real", type=str, default="C:/Users/yons/Desktop/CODE/AAAI_2022/datasets/real_input", help='path to training data of real rain')

parser.add_argument("--use_contrast", type=bool, default=True, help='use contrasive regularization or not')
parser.add_argument("--use_stage1", type=bool, default=True, help='use stage1: train on synthesize image or not')
parser.add_argument("--use_stage2", type=bool, default=True, help='use stage2: train on real image or not')

parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--num_of_SegClass", type=int, default=21, help='Number of Segmentation Classes, default VOC = 21')

opt = parser.parse_args()

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

################## USRA block starts here ################
def SegLoss(input_train, out_train):
    # TODO: (1) Seg is too large, consider scale with 0.1  (2) Try Segment the Image, not the rain residual
    # Build Segmentation model
    seg = fpn(opt.num_of_SegClass).to(device)
    seg_criterion = FocalLoss(gamma=2).to(device)

    # Build and clip residual image
    out_train = torch.clamp(input_train - out_train, 0., 1.)

    # Build and dmean seg. input (maybe clip image before)
    seg_input = out_train.data.cpu().numpy()
    for n in range(out_train.size()[0]):
        seg_input[n, :, :, :] = rgb_demean(seg_input[n, :, :, :])

    # send seg. input to cuda
    seg_input = Variable(torch.from_numpy(seg_input).to(device))

    # build seg. output
    seg_output = seg(seg_input)

    # build seg. target
    target = (get_NoGT_target(seg_output)).data.cpu()
    target_ = resize_target(target, seg_output.size(2))
    target_ = torch.from_numpy(target_).long().to(device)

    # calculate seg. loss
    seg_loss = seg_criterion(seg_output, target_).to(device)

    # freeze seg. backpropagation
    for param in seg.parameters():
        param.requires_grad = False

    return seg_loss

data_transfrom = transforms.Compose([
        transforms.RandomCrop(100, 100),
        transforms.ToTensor()
    ])


def main():
    print('Loading Synthetic Rainy dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train,
                              num_workers=0,
                              batch_size=opt.batch_size,
                              shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    print('Loading Real Rainy dataset ...\n')
    img = datasets.ImageFolder(opt.data_path_real,
                               transform=data_transfrom)
    imgLoader = DataLoader(dataset=img,
                           batch_size=opt.batch_size,
                           shuffle=True)

    # Build deraining model
    model = SAPNet(recurrent_iter=opt.recurrent_iter,
                   use_Contrast=opt.use_contrast).to(device)
    print_network(model)

    # Define SSIM
    criterion = SSIM().to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # Start training
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        if opt.use_stage1:
            # Phase 1 Training (Synthetic images)
            for i, (input_train, target_train) in enumerate(loader_train, 0):
                model.train()
                model.zero_grad()
                optimizer.zero_grad()

                input_train, target_train = Variable(input_train).to(device), Variable(target_train).to(device)

                # Obtain the derained image and calculate ssim loss
                out_train, _ = model(input_train)

                #print("input_train", input_train.size()) # torch.Size([batch_size, 3, 100, 100])
                #print("target_train", target_train.size()) # torch.Size([batch_size, 3, 100, 100])
                #print("out_train", out_train.size()) # torch.Size([batch_size, 3, 100, 100])

                pixel_metric = criterion(target_train, out_train)

                # Negative SSIM loss
                loss = -pixel_metric

                # backward and update parameters.
                loss.backward()
                optimizer.step()

                ####### Right now Keep this part unmodified
                model.eval()
                out_train, _ = model(input_train)
                out_train = torch.clamp(out_train, 0., 1.)
                psnr_train = batch_PSNR(out_train, target_train, 1.)
                print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                      (epoch + 1, i + 1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

        if opt.use_stage2:
            # Phase 2 Training (Real images)
            for i, (input_train_real, _) in enumerate(imgLoader, 0):
                model.train()
                model.zero_grad()
                optimizer.zero_grad()

                input_train_real = Variable(input_train_real).to(device)

                # Obtain the derained image and calculate ssim loss
                out_train_real, _ = model(input_train_real)

                #print("input_train", input_train_real.size())
                #print("out_train", out_train_real.size())

                # Segmentation loss
                seg_loss = SegLoss(input_train_real, out_train_real)

                # backward and update parameters.
                seg_loss.backward()
                optimizer.step()

                print("[epoch %d][%d/%d] loss: %.4f" %
                      (epoch + 1, i + 1, len(imgLoader), seg_loss.item()))

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            print(opt.data_path.find('RainTrainH'))
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')


    main()
