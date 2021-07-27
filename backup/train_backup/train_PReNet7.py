# Purpose: to combine URSS, Canny and channel attention

import argparse
import torch.optim as optim
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from Modeling.DerainDataset import *
from Modeling.utils import *
from torch.optim.lr_scheduler import MultiStepLR
from Modeling.SSIM import SSIM
from Modeling.network_tmp.networks7 import *
from Modeling.fpn import *


parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30, 50, 80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/PReNet_test/Model7", help='path to save models and log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--data_path", type=str, default="datasets/train/RainTrainH", help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=False, help='use GPU or not')  # Set True for GPU training
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
# add for segmentation model
parser.add_argument("--num_of_SegClass", type=int, default=21, help='Number of Segmentation Classes, default VOC = 21')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0,
                              batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build deraining model
    model = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    print_network(model)

    # Build Segmentation model
    seg = fpn(opt.num_of_SegClass)
    seg_criterion = FocalLoss(gamma=2)
    # seg = nn.DataParallel(seg)

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()

    # Move models and criterions to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()
        seg_criterion = seg_criterion.cuda()
        seg = seg.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        # epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            # Obtain the derained image and calculate ssim loss
            #out_train, _ = model(input_train)
            out_train, _ = model(input_train)

            pixel_metric = criterion(target_train, out_train)

            ################### Calculating edges starts here ###################
            # Edge for clean image
            target_train_edge = Laplacian(target_train)

            # Edge for derained image
            out_train_edge = Laplacian(out_train)

            # Edge loss
            edge_loss = inference_mse_loss(target_train_edge, out_train_edge)

            # SSIM and edge loss
            loss = -pixel_metric + 0.05 * edge_loss

            #loss = -pixel_metric

            ################## USRA block starts here ################
            # Build and clip residual image
            out_train = torch.clamp(input_train - out_train, 0., 1.)

            # Build and dmean seg. input (maybe clip image before)
            seg_input = out_train.data.cpu().numpy()
            for n in range(out_train.size()[0]):
                seg_input[n, :, :, :] = rgb_demean(seg_input[n, :, :, :])

            # send seg. input to cuda
            if opt.use_gpu:
                seg_input = Variable(torch.from_numpy(seg_input).cuda())
            else:
                seg_input = Variable(torch.from_numpy(seg_input))

            # build seg. output
            seg_output = seg(seg_input)

            # build seg. target
            target = (get_NoGT_target(seg_output)).data.cpu()
            target_ = resize_target(target, seg_output.size(2))
            target_ = torch.from_numpy(target_).long()

            if opt.use_gpu:
                target_ = target_.cuda()

            # calculate seg. loss
            seg_loss = seg_criterion(seg_output, target_)

            # freeze seg. backpropagation
            for param in seg.parameters():
                param.requires_grad = False

            # calculate total loss
            total_loss = loss + seg_loss

            # backward and update parameters.
            total_loss.backward()
            optimizer.step()

            ####### Right now Keep this part unmodified

            # training curve
            model.eval()
            #out_train, _ = model(input_train)
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            #print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  #(epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), total_loss.item(), pixel_metric.item(), psnr_train))

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', total_loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        # epoch training end

        # log the images
        model.eval()
        #out_train, _ = model(input_train)
        out_train, _ = model(input_train)
        out_train = torch.clamp(out_train, 0., 1.)
        im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', im_target, epoch+1)
        writer.add_image('rainy image', im_input, epoch+1)
        writer.add_image('deraining image', im_derain, epoch+1)

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
