import torch
import torch.nn as nn
import numpy as np
import math


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class RecursiveBlock(nn.Module):
    def __init__(self, d):
        super(RecursiveBlock, self).__init__()

        self.block = nn.Sequential()
        for i in range(d):
            self.block.add_module("conv_" + str(i), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                                                              stride=1, padding=1, bias=True))
            self.block.add_module("relu_" + str(i), nn.LeakyReLU(0.2, inplace=True))

        for m in self.block.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        output = self.block(x)
        return output


class FeatureEmbedding(nn.Module):
    def __init__(self, r, d):
        super(FeatureEmbedding, self).__init__()

        self.recursive_block = RecursiveBlock(d)
        self.num_recursion = r

    def forward(self, x):
        output = x.clone()

        # The weights are shared within the recursive block!
        for i in range(self.num_recursion):
            output = self.recursive_block(output) + x

        return output


class LapSrnMS(nn.Module):
    def __init__(self, r, d, scale):
        super(LapSrnMS, self).__init__()

        self.scale = scale
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4,
                                            stride=2, padding=1, bias=True)
        self.relu_features = nn.LeakyReLU(0.2, inplace=True)

        self.scale_img = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4,
                                            stride=2, padding=1, bias=True)
        self.predict = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        self.features = FeatureEmbedding(r, d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)

                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        features = self.relu(self.conv_input(x))
        output_images = []
        rescaled_img = x.clone()

        for i in range(int(math.log2(self.scale))):
            features = self.features(features)
            features = self.relu_features(self.transpose(features))

            rescaled_img = self.scale_img(rescaled_img)
            predict = self.predict(features)
            out = torch.add(predict, rescaled_img)

            output_images.append(out)

        return output_images


class CharbonnierLoss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        # print(error)
        loss = torch.sum(error)
        return loss
