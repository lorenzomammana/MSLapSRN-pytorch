import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image


def transform(img_lr, img_2x, img_4x, settype):
    # Resize
    if settype == "train":
        # Random horizontal flipping
        if random.random() > 0.5:
            img_lr = tf.hflip(img_lr)
            img_2x = tf.hflip(img_2x)
            img_4x = tf.hflip(img_4x)

        # Random horizontal flipping
        if random.random() > 0.5:
            img_lr = tf.vflip(img_lr)
            img_2x = tf.vflip(img_2x)
            img_4x = tf.vflip(img_4x)

        # Random rotation
        rotations = [0, 90, 180, 270]
        pick_rotation = rotations[random.randint(0, 3)]
        img_lr = tf.rotate(img_lr, pick_rotation)
        img_2x = tf.rotate(img_2x, pick_rotation)
        img_4x = tf.rotate(img_4x, pick_rotation)

        resize_factor = random.uniform(0.5, 1)
        img_lr = img_lr.resize((int(img_lr.size[0] * resize_factor), int(img_lr.size[1] * resize_factor)), Image.BICUBIC)
        img_2x = img_2x.resize((int(img_2x.size[0] * resize_factor), int(img_2x.size[1] * resize_factor)), Image.BICUBIC)
        img_4x = img_4x.resize((int(img_4x.size[0] * resize_factor), int(img_4x.size[1] * resize_factor)), Image.BICUBIC)

        # Transform to tensor
        img_lr = tf.to_tensor(transforms.Resize((32, 32), Image.BICUBIC)(img_lr))
        img_2x = tf.to_tensor(transforms.Resize((64, 64), Image.BICUBIC)(img_2x))
        img_4x = tf.to_tensor(transforms.Resize((128, 128), Image.BICUBIC)(img_4x))
    else:
        img_lr = tf.to_tensor(img_lr)
        img_2x = tf.to_tensor(img_2x)
        img_4x = tf.to_tensor(img_4x)

    return img_lr, img_2x, img_4x


class SRdataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_ids, settype):
        """Initialization"""
        with open(list_ids, 'r') as f:
            self.list_ids = f.read().splitlines()
            self.settype = settype

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        id = self.list_ids[index]

        # Load data and get label
        img_4x = Image.open('dataset/{}_patches/4x/{}'.format(self.settype, id))
        img_2x = Image.open('dataset/{}_patches/2x/{}'.format(self.settype, id))
        img_lr = Image.open('dataset/{}_patches/lr/{}'.format(self.settype, id))
        img_4x = img_4x.convert('YCbCr')
        img_2x = img_2x.convert('YCbCr')
        img_lr = img_lr.convert('YCbCr')

        return transform(img_lr.getchannel(0),
                         img_2x.getchannel(0),
                         img_4x.getchannel(0),
                         self.settype)
