import torch
import numpy as np
import time, math, glob
from lapsrn import LapSrnMS
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


def PSNR(pred, gt, shave_border=0):
    gt = transforms.ToTensor()(gt)
    if isinstance(pred, Image.Image):
        pred = transforms.ToTensor()(pred)

    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = torch.sqrt(torch.mean(torch.pow(imdff, 2)))
    if rmse == 0:
        return 100
    return 20 * torch.log10(gt.max() / rmse)


cuda = True

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

checkpoint = torch.load('best.pt')

model = LapSrnMS(5, 5, 4)
model.load_state_dict(checkpoint['state_dict'])

dataset_list = glob.glob('dataset/test/*/')

if cuda:
    model = model.to('cuda:2')
else:
    model = model.cpu()
model.eval()


for dataset in dataset_list:
    avg_psnr_predicted_2x = 0.0
    avg_psnr_bicubic_2x = 0.0
    avg_psnr_predicted_4x = 0.0
    avg_psnr_bicubic_4x = 0.0
    avg_elapsed_time = 0.0
    image_list = glob.glob(dataset + '*.png')

    for image_name in image_list:
        im_hr = Image.open(image_name)
        im_hr_ycbcr = im_hr.convert('YCbCr')
        
        im_1_2x = im_hr_ycbcr.resize((int(im_hr.size[0] / 2), int(im_hr.size[1] / 2)), Image.BICUBIC)
        im_1_4x = im_hr_ycbcr.resize((int(im_hr.size[0] / 4), int(im_hr.size[1] / 4)), Image.BICUBIC)

        im_2x_b = im_1_2x.resize((im_hr.size[0], im_hr.size[1]), Image.BICUBIC)
        im_4x_b = im_1_4x.resize((im_hr.size[0], im_hr.size[1]), Image.BICUBIC)

        psnr_bicubic_2x = PSNR(im_hr_ycbcr.getchannel(0), im_2x_b.getchannel(0), shave_border=0)
        psnr_bicubic_4x = PSNR(im_hr_ycbcr.getchannel(0), im_4x_b.getchannel(0), shave_border=0)
        avg_psnr_bicubic_2x += psnr_bicubic_2x
        avg_psnr_bicubic_4x += psnr_bicubic_4x

        # Process 2x
        im_input = transforms.ToTensor()(im_1_2x.getchannel(0))
        im_input = im_input.float()
        im_input = im_input.unsqueeze(0)

        if cuda:
            im_input = im_input.to('cuda:2')

        start_time = time.time()
        HR_2x, _ = model(im_input)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        HR_2x = HR_2x.cpu()

        im_2x_m = HR_2x

        im_2x_m[im_2x_m < 0] = 0
        im_2x_m[im_2x_m > 1] = 1
        im_2x_m = im_2x_m[0, :, :]

        im_2x_m = transforms.ToPILImage()(im_2x_m)

        if im_2x_m.size != im_hr.size:
            im_2x_m = im_2x_m.resize(im_hr.size, Image.BICUBIC)

        im_2x_m = Image.merge('YCbCr', [im_2x_m, im_2x_b.getchannel(1), im_2x_b.getchannel(2)])

        rgb_out = im_2x_m.convert('RGB')
        rgb_out.save('output/' + image_name.split('/')[-2] + '/2x/' + image_name.split('/')[-1], 'PNG')

        psnr_predicted = PSNR(im_hr_ycbcr.getchannel(0), im_2x_m.getchannel(0), shave_border=0)
        avg_psnr_predicted_2x += psnr_predicted

        # Process 4x
        im_input = transforms.ToTensor()(im_1_4x.getchannel(0))
        im_input = im_input.float()
        im_input = im_input.unsqueeze(0)

        if cuda:
            im_input = im_input.to('cuda:2')

        start_time = time.time()
        _, HR_4x = model(im_input)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        HR_4x = HR_4x.cpu()

        im_4x_m = HR_4x

        im_4x_m[im_4x_m < 0] = 0
        im_4x_m[im_4x_m > 1] = 1
        im_4x_m = im_4x_m[0, :, :]

        im_4x_m = transforms.ToPILImage()(im_4x_m)

        if im_4x_m.size != im_hr.size:
            im_4x_m = im_4x_m.resize(im_hr.size, Image.BICUBIC)

        im_4x_m = Image.merge('YCbCr', [im_4x_m, im_4x_b.getchannel(1), im_4x_b.getchannel(2)])

        rgb_out = im_4x_m.convert('RGB')
        rgb_out.save('output/' + image_name.split('/')[-2] + '/4x/' + image_name.split('/')[-1], 'PNG')

        psnr_predicted = PSNR(im_hr_ycbcr.getchannel(0), im_4x_m.getchannel(0), shave_border=0)
        avg_psnr_predicted_4x += psnr_predicted

    print("Dataset=", dataset)
    print("PSNR_predicted_2x=", avg_psnr_predicted_2x / len(image_list))
    print("PSNR_bicubic_2x=", avg_psnr_bicubic_2x / len(image_list))
    print("PSNR_predicted_4x=", avg_psnr_predicted_4x / len(image_list))
    print("PSNR_bicubic_4x=", avg_psnr_bicubic_4x / len(image_list))
