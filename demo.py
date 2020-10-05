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


checkpoint = torch.load('best.pt', map_location='cuda:0')
net = LapSrnMS(5, 5, 8)
net.load_state_dict(checkpoint['state_dict'])
net.to('cuda:0')

input_img = Image.open("graffiti-508272__340.jpg")
input_img = input_img.convert('YCbCr')
im_y = input_img.getchannel(0)
im_cb = input_img.getchannel(1)
im_cr = input_img.getchannel(2)

im = tf.to_tensor(im_y)
im = im.unsqueeze(0)
im = im.to('cuda:0')

with torch.no_grad():
    out_2x, out_4x, out_8x = net(im)

out_2x = transforms.ToPILImage()(out_2x[0].cpu())
out_4x = transforms.ToPILImage()(out_4x[0].cpu())
out_8x = transforms.ToPILImage()(out_8x[0].cpu())

im_cb_2x = im_cb.resize((im_cb.size[0] * 2, im_cb.size[1] * 2), Image.BICUBIC)
im_cb_4x = im_cb.resize((im_cb.size[0] * 4, im_cb.size[1] * 4), Image.BICUBIC)
im_cb_8x = im_cb.resize((im_cb.size[0] * 8, im_cb.size[1] * 8), Image.BICUBIC)

im_cr_2x = im_cr.resize((im_cr.size[0] * 2, im_cr.size[1] * 2), Image.BICUBIC)
im_cr_4x = im_cr.resize((im_cr.size[0] * 4, im_cr.size[1] * 4), Image.BICUBIC)
im_cr_8x = im_cr.resize((im_cr.size[0] * 8, im_cr.size[1] * 8), Image.BICUBIC)

out_2x = Image.merge('YCbCr', [out_2x, im_cb_2x, im_cr_2x])
out_4x = Image.merge('YCbCr', [out_4x, im_cb_4x, im_cr_4x])
out_8x = Image.merge('YCbCr', [out_8x, im_cb_8x, im_cr_8x])

out_2x = out_2x.convert('RGB')
out_4x = out_4x.convert('RGB')
out_8x = out_8x.convert('RGB')

out_2x.save("out_2x.png", "PNG")
out_4x.save("out_4x.png", "PNG")
out_8x.save("out_8x.png", "PNG")
