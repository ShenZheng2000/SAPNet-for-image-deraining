import torch.optim as optim
from torch.utils.data import DataLoader
from Modeling.DerainDataset import *
from Modeling.utils import *
from torch.optim.lr_scheduler import MultiStepLR
from Modeling.SSIM import SSIM
from Modeling.network import *
from Modeling.fpn import *
from torchvision import datasets, transforms
from option import *
from loss_fun import *
import torch.nn as nn

def SegLoss(out_train):
    num_of_SegClass = 21
    seg = fpn(num_of_SegClass).to(device)
    seg_criterion = FocalLoss(gamma=2).to(device)

    # build seg. output
    seg_output = seg(out_train).to(device)

    # build seg. target
    target = (get_NoGT_target(seg_output)).to(device)

    # Get seg. loss
    seg_loss = seg_criterion(seg_output, target).to(device)

    # freeze seg. backpropagation
    for param in seg.parameters():
        param.requires_grad = False

    return seg_loss



def train():
    data_transfrom = transforms.Compose([
        transforms.RandomCrop(100, 100),
        transforms.ToTensor()
    ])

    '''if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id'''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [Id for Id in range(torch.cuda.device_count())]

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
    model = SAPNet(recurrent_iter=opt.recurrent_iter, use_dilation=opt.use_dilation).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    print_network(model)

    # Define SSIM and constrative loss
    criterion = SSIM().to(device)
    loss_C = ContrastLoss().to(device)

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
                loss_ssim = -pixel_metric
                #print("loss_ssim", loss_ssim)

                # Constrative loss
                loss_contrast = loss_C(out_train, target_train, input_train) if opt.use_contrast else 0
                #print("loss_contrast", loss_contrast)

                # Total loss
                loss = loss_ssim + loss_contrast

                # backward and update parameters.
                loss.backward()
                optimizer.step()

                ####### Right now Keep this part unmodified
                model.eval()
                out_train, _ = model(input_train)
                out_train = torch.clamp(out_train, 0., 1.)
                psnr_train = batch_PSNR(out_train, target_train, 1.)
                if i % 50 == 0:
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

                # Segmentation loss
                seg_loss = SegLoss(out_train_real)

                # backward and update parameters.
                seg_loss.backward()
                optimizer.step()

                if i % 50 == 0:
                    print("[epoch %d][%d/%d] loss: %.4f" %
                      (epoch + 1, i + 1, len(imgLoader), seg_loss.item()))

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    train()
