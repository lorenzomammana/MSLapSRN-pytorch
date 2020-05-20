import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image


def transform(img, settype):
    # Resize
    if settype == "train":
        # Random horizontal flipping
        if random.random() > 0.5:
            img = tf.hflip(img)

        # Random rotation
        rotations = [0, 90, 180, 270]
        pick_rotation = rotations[random.randint(0, 3)]
        img = tf.rotate(img, pick_rotation)

        # Transform to tensor
        img_lr = tf.to_tensor(transforms.Resize((32, 32), Image.BICUBIC)(img))
        img_2x = tf.to_tensor(transforms.Resize((64, 64), Image.BICUBIC)(img))
        img_4x = tf.to_tensor(img)
    else:
        img_lr = tf.to_tensor(transforms.Resize((32, 32), Image.BICUBIC)(img))
        img_2x = tf.to_tensor(transforms.Resize((64, 64), Image.BICUBIC)(img))
        img_4x = tf.to_tensor(img)

    return img_lr, img_2x, img_4x


class SRdataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_ids, settype):
        """Initialization"""
        with open(list_ids, 'r') as f:
            self.list_ids = f.read().splitlines()
            self.settype = settype
            self.patch_size = 128
            self.eps = 1e-3

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        id = self.list_ids[index]

        # Load data and get label
        img = Image.open('dataset/{}/{}'.format(self.settype, id))
        img = img.convert('YCbCr')
        img = img.getchannel(0)

        if self.settype == 'train':
            resize_factor = random.uniform(0.5, 1)

            if img.size[0] < img.size[1]:
                if img.size[0] * resize_factor < self.patch_size:
                    resize_factor = self.patch_size / img.size[0] + self.eps
            else:
                if img.size[1] * resize_factor < self.patch_size:
                    resize_factor = self.patch_size / img.size[1] + self.eps

            img = img.resize((int(img.size[0] * resize_factor), int(img.size[1] * resize_factor)), Image.BICUBIC)

            print(img.size)
            img = transforms.RandomCrop((self.patch_size, self.patch_size))(img)
        else:
            img = img.resize(self.patch_size, self.patch_size, Image.BICUBIC)

        return transform(img, self.settype)
