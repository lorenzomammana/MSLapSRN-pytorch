import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
from ssim import ssim

from lapsrn import LapSrnMS


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def SSIM(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    return np.mean(ssim(pred, gt))


if __name__ == '__main__':
    cuda = True
    checkpoint = torch.load('best.pt')

    model = LapSrnMS(5, 5, 4)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to('cuda:2')
    model.eval()

    for scale in [2, 4]:
        for dataset in glob.glob('dataset/mat/*/'):
            image_list = glob.glob('{}{}x/*.mat'.format(dataset, scale))

            avg_psnr_predicted = 0.0
            avg_psnr_bicubic = 0.0
            avg_ssim_predicted = 0.0
            avg_ssim_bicubic = 0.0
            avg_elapsed_time = 0.0

            for image_name in image_list:
                # print("Processing ", image_name)
                im_gt_y = sio.loadmat(image_name)['im_gt_y']
                im_b_y = sio.loadmat(image_name)['im_b_y']
                im_l_y = sio.loadmat(image_name)['im_l_y']

                im_gt_y = im_gt_y.astype(float)
                im_b_y = im_b_y.astype(float)
                im_l_y = im_l_y.astype(float)

                psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=scale)
                avg_psnr_bicubic += psnr_bicubic
                avg_ssim_bicubic += SSIM(im_gt_y, im_b_y, shave_border=scale)

                im_input = im_l_y / 255.

                im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

                if cuda:
                    im_input = im_input.to('cuda:2')
                else:
                    model = model.cpu()

                start_time = time.time()
                if scale == 2:
                    HR_4x, _ = model(im_input)
                if scale == 4:
                    _, HR_4x = model(im_input)
                elapsed_time = time.time() - start_time
                avg_elapsed_time += elapsed_time

                HR_4x = HR_4x.cpu()

                im_h_y = HR_4x.data[0].numpy().astype(np.float32)

                im_h_y = im_h_y * 255.
                im_h_y[im_h_y < 0] = 0
                im_h_y[im_h_y > 255.] = 255.
                im_h_y = im_h_y[0, :, :]

                psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)

                avg_psnr_predicted += psnr_predicted
                avg_ssim_predicted += SSIM(im_gt_y, im_h_y, shave_border=scale)

            print("Scale=", scale)
            print("Dataset=", dataset)
            print("PSNR_predicted=", avg_psnr_predicted / len(image_list))
            print("PSNR_bicubic=", avg_psnr_bicubic / len(image_list))
            print("SSIM_predicted=", avg_ssim_predicted / len(image_list))
            print("SSIM_bicubic=", avg_ssim_bicubic / len(image_list))
            print("It takes average {}s for processing".format(avg_elapsed_time / len(image_list)))
