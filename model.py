from math import log2
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Generator(nn.Module):
    def __init__(self, img_shape: Tuple[int], generator_params: dict, generator_type: str = 'dcnn',
                 latent_dim: int = 100):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.generator_type = generator_type
        if generator_type == 'dcnn':
            self.GenConvModel = nn.Sequential()
            self.build_dcnn_generator_v2(**generator_params)

    def build_dcnn_generator(self, z_shape: Tuple[int], nlastfilters: int = 64,
                             filter_size: Tuple[int] = (3, 3), lrelu_coef: float = 0.2):
        img_width = self.img_shape[1]
        layers_cnt = int(log2(img_width) - log2(z_shape[1]))
        padding = int((filter_size[0] - 1) / 2)

        infilters = 0
        for i in range(layers_cnt):
            if infilters == 0:
                infilters = z_shape[0]
            else:
                infilters = nlastfilters * pow(2, layers_cnt - i)
            outfilters = nlastfilters * pow(2, layers_cnt - i - 1)

            infilters = min(infilters, 256)
            outfilters = min(outfilters, 256)

            self.GenConvModel.add_module(f'ConvTranspose_{i}', nn.ConvTranspose2d(infilters, outfilters, filter_size, 2,
                                                                                  padding, output_padding=1))
            self.GenConvModel.add_module(f'BN_{i}', nn.BatchNorm2d(outfilters))
            self.GenConvModel.add_module(f'LReLU_{i}', nn.LeakyReLU(lrelu_coef, True))

            self.GenConvModel.add_module(f'Conv_{i}',
                                         nn.Conv2d(outfilters, outfilters, filter_size, padding=padding))
            self.GenConvModel.add_module(f'BN_{i}', nn.BatchNorm2d(outfilters))
            self.GenConvModel.add_module(f'LReLU_{i}', nn.LeakyReLU(lrelu_coef, True))

        self.GenConvModel.add_module('FinalConvTranspose',
                                     nn.ConvTranspose2d(nlastfilters, self.img_shape[0], filter_size, 1, padding))
        self.GenConvModel.add_module('Tanh', nn.Tanh())

    def build_dcnn_generator_v2(self, z_shape: Tuple[int], init_filters: int = 1024,
                             filter_size: Tuple[int] = (3, 3), lrelu_coef: float = 0.2):
        img_width = self.img_shape[1]
        layers_cnt = int(log2(img_width))
        padding = int((filter_size[0] - 1) / 2)

        infilters = 0
        outfilters = 0
        for i in range(layers_cnt):
            if infilters == 0:
                infilters = z_shape[0]
            else:
                infilters = outfilters
            if outfilters == 0:
                outfilters = init_filters
            else:
                outfilters = int(infilters/2)

            self.GenConvModel.add_module(f'ConvTranspose_{i}', nn.ConvTranspose2d(infilters, outfilters, filter_size, 2,
                                                                                  padding, output_padding=1))
            self.GenConvModel.add_module(f'BN_{i}', nn.BatchNorm2d(outfilters))
            self.GenConvModel.add_module(f'LReLU_{i}', nn.LeakyReLU(lrelu_coef, True))


        self.GenConvModel.add_module('FinalConvTranspose',
                                     nn.ConvTranspose2d(outfilters, self.img_shape[0], filter_size, 1, padding))
        self.GenConvModel.add_module('Tanh', nn.Tanh())

    def forward(self, x):
        x = self.GenConvModel(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, img_shape: Tuple[int, ...], generator_params: dict,
                 generator_type: str = 'dcnn_classes'):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        if generator_type == 'dcnn_classes':
            self.build_dcnn_classes_discriminator(**generator_params)

    def build_dcnn_classes_discriminator(self, num_classes: int, init_filter_cnt: int = 64, lrelu_coef: float = 0.2,
                                         conv_cnt: int = 6, filter_size: Tuple[int] = (3, 3), drop_rate: float = 0.25):
        self.n_classes = num_classes
        self.nconv = conv_cnt
        padding = int((filter_size[0] - 1) / 2)

        self.Featurizer = nn.Sequential()

        self.Featurizer.add_module(f'FeaturizerConv_0',
                                   nn.Conv2d(self.img_shape[0], init_filter_cnt, filter_size, 2, padding))
        self.Featurizer.add_module(f'FeaturizerLReLU_0', nn.LeakyReLU(lrelu_coef))

        outfilters = init_filter_cnt * 2
        infilters = init_filter_cnt
        for i in range(self.nconv - 1):
            self.Featurizer.add_module(f'FeaturizerConv_{i + 1}',
                                       nn.Conv2d(infilters, outfilters, filter_size, 2, padding))
            self.Featurizer.add_module(f'FeaturizerBN_{i + 1}', nn.BatchNorm2d(outfilters))
            self.Featurizer.add_module(f'FeaturizerLReLU_{i + 1}', nn.LeakyReLU(lrelu_coef))

            infilters = outfilters
            outfilters *= 2

        out_size = int(self.img_shape[1] / pow(2, conv_cnt))
        nfeats = infilters * out_size * out_size
        self.ValidModel = nn.Sequential(
            nn.Linear(nfeats, 1)
        )
        self.ClassModel = nn.Sequential(
            nn.Linear(nfeats, 256),
            nn.LeakyReLU(lrelu_coef),
            nn.Linear(256, 128),
            nn.LeakyReLU(lrelu_coef),
            nn.Linear(128, self.n_classes)
        )

    def forward(self, x):
        x = self.Featurizer(x)
        x = x.view(x.shape[0], -1)


        valid = self.ValidModel(x)
        classification = self.ClassModel(x)

        return valid, classification
