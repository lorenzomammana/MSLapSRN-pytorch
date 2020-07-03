from lapsrn import *
from PIL import Image, ImageFilter
import torchvision.transforms.functional as tf
from torchvision import transforms


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


def get_y(img):
    img = img.convert('YCbCr')
    img = img.getchannel(0)

    return img


checkpoint = torch.load('best.pt')
net = LapSrnMS(5, 5, 4)
net.load_state_dict(checkpoint['state_dict'])
net.to('cuda:2')

im_4x = get_y(Image.open("../ir-lapsrn/dataset/FLIR/test/registered-rgb/FLIR_video_03273.jpeg"))

im = tf.to_tensor(im_4x)
im = im.unsqueeze(0)
im = im.to('cuda:2')

with torch.no_grad():
    out_2x, out_4x = net(im)
    out_2x[out_2x > 1] = 1
    out_4x[out_4x > 1] = 1

out_2x = transforms.ToPILImage()(out_2x[0].cpu())
out_4x = transforms.ToPILImage()(out_4x[0].cpu())

out_2x.save("out_2x.png", "PNG")
out_4x.save("out_4x.png", "PNG")