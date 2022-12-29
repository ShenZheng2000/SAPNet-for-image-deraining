from Modeling.DerainDataset import *
from Modeling.utils import *
from Modeling.network import *
import time
from option import *

def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [Id for Id in range(torch.cuda.device_count())]

    os.makedirs(opt.output_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = SAPNet(recurrent_iter=opt.recurrent_iter,
                   use_dilation=opt.use_dilation).to(device)
    print_network(model)
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_latest.pth'), map_location=device))
    model.eval()

    time_test = 0
    count = 0
    for img_name in os.listdir(opt.test_data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.test_data_path, img_name)

            # input image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y)).to(device)

            with torch.no_grad(): #
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()

                out, _ = model(y)
                out = torch.clamp(out, 0., 1.)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                # print(img_name, ': ', dur_time)

            if torch.cuda.is_available():
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.output_path, img_name), save_out)

            count += 1

    print('Avg. time:', time_test/count)


if __name__ == "__main__":
    test()
