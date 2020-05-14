from torch.utils import data
from SRdataset import SRdataset
from lapsrn import *

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 4}

# Generators
training_set = SRdataset("train_patches.txt")
training_generator = data.DataLoader(training_set, **params)