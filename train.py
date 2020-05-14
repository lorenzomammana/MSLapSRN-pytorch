from torch.utils import data
import torch.optim as optim
from SRdataset import SRdataset
from lapsrn import *
import shutil


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=100):
    """Decay learning rate by a factor of 2 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 4}
max_epochs = 1000

# Generators
training_set = SRdataset("train_patches.txt")
training_generator = data.DataLoader(training_set, **params)

net = LapSrnMS(5, 5, 4)

if use_cuda:
    net.to(device)

criterion = CharbonnierLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

if __name__ == '__main__':
    # Loop over epochs
    loss_min = np.inf
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=1e-4, lr_decay_epoch=100)
        running_loss = 0.0

        for i, data in enumerate(training_generator, 0):
            # get the inputs; data is a list of [inputs, labels]
            in_lr, in_2x, in_4x = data[0].to(device), data[1].to(device), data[2].to(device)

            in_lr.requires_grad = True
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            out_2x, out_4x = net(in_lr)
            loss_2x = criterion(in_2x, out_2x)
            loss_4x = criterion(in_4x, out_4x)

            loss = (loss_2x + loss_4x) / in_lr.shape[0]

            loss_2x.backward(retain_graph=True)

            loss_4x.backward()

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)

            optimizer.step()

            if loss.item() < loss_min:
                checkpoint = {
                    'epoch': epoch + 1,
                    'valid_loss_min': loss,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_ckp(checkpoint, True, "ckp.pt", "best.pt")
                loss_min = loss.item()
                print("Best model loss: {}".format(loss))

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')