from torch.utils import data
from SRdataset import SRdataset
from torchvision import transforms

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 4}

# Generators
training_set = SRdataset("train")
training_generator = data.DataLoader(training_set, **params)

for i, data in enumerate(training_generator):
    in_lr, in_2x, in_4x, in_8x = transforms.ToPILImage()(data[0].squeeze()),\
                                 transforms.ToPILImage()(data[1].squeeze()),\
                                 transforms.ToPILImage()(data[2].squeeze()), transforms.ToPILImage()(data[3].squeeze())
    in_lr.save('testing/{}_lr.png'.format(i))
    in_2x.save('testing/{}_2x.png'.format(i))
    in_4x.save('testing/{}_4x.png'.format(i))
    in_8x.save('testing/{}_8x.png'.format(i))
    if i == 10:
        break
