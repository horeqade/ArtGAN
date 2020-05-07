from datetime import date
from os import getcwd, makedirs
from os.path import join, exists
from typing import Tuple, List
import shutil

import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
from sklearn.preprocessing import LabelBinarizer
from torch.autograd import Variable
from torchsummary import summary

from ArtistAI.data_processor import get_data, get_images_classes, combine_images, get_one_image
from ArtistAI.model import Generator, Discriminator
from ArtistAI.util import model_saver

ImageFile.LOAD_TRUNCATED_IMAGES = True
save_path = join(getcwd(), 'saved_model')


class WasserteineLoss(torch.nn.Module):
    def __init__(self):
        super(WasserteineLoss, self).__init__()

    def forward(self, predict, target):
        pass


class TrainerGAN:
    def __init__(self, data_path: str, missing_folders: List[str] = [], folders: List[str] = None):
        self.data, self.num_styles, self.classes = get_data(data_path, missing_folders, folders)
        self.label_bin = LabelBinarizer()
        self.label_bin.fit(list(range(self.num_styles)))
        self.img_shape = (3, int(data_path.split('images')[1]), int(data_path.split('images')[1]))

        self.netG = self.netD = self.optimizer_g = self.optimizer_d = None
        self.epoch_ini = 0

    def init_model(self, gen_params: dict, discr_params: dict, optimizator_params: dict,
                   month_day: str = None, epoch_ini: int = None):
        self.z_shape = gen_params['z_shape']
        discr_params['num_classes'] = self.num_styles
        self.netG = Generator(self.img_shape, gen_params)
        self.netD = Discriminator(self.img_shape, discr_params)
        self.optimizer_g, self.optimizer_d = self.load_optimizer_(optimizator_params)
        self.loss_d_label = torch.nn.CrossEntropyLoss()

        cuda = torch.cuda.is_available()
        if cuda:
            self.netG.cuda()
            self.netD.cuda()

        print(summary(self.netD, img_shape))
        print(summary(self.netG, self.z_shape))

        self.netG.apply(self.weights_init_normal_)
        self.netD.apply(self.weights_init_normal_)

    def train(self, epochs: int = 100, batch_size: int = 4, gif: bool = False):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        gif_i = 0
        if not gif:
            save_folder = 'generated'
        else:
            save_folder = 'generated_gif'

        for epoch in range(self.epoch_ini, epochs):
            if epoch % 5 == 0:
                self.model_saver_(epoch)
            for index in range(int(len(self.data) / batch_size)):
                # real_images, real_labels = get_images_classes(BATCH_SIZE, data, classes, img_shape,
                #                                              batch_num=index, channel_first=True, )

                real_images, real_labels = get_one_image(batch_size, self.data, self.classes, self.img_shape,
                                                         channel_first=True, img_idx=6)

                real_images = Variable(torch.from_numpy(real_images).cuda().type(torch.float32))
                real_labels = Variable(torch.from_numpy(real_labels).cuda().type(torch.long))

                self.optimizer_d.zero_grad()
                if index % 2 == 0:
                    # Train on real images

                    d_validation, d_class = self.netD(real_images)
                    d_real_loss = -torch.mean(d_validation)

                    if self.num_styles > 1:
                        d_label_real_loss = self.loss_d_label(d_class, real_labels)
                        print(f"epoch {epoch} batch {index} D_validity_loss : {d_real_loss}, ",
                              f"D_label_loss: {d_label_real_loss}")
                    else:
                        print(f'epoch {epoch} batch {index} d_loss : {d_real_loss}')

                    d_total_real_loss = d_real_loss + d_label_real_loss
                    d_total_real_loss.backward()
                    self.optimizer_d.step()

                else:
                    # Train on artificial images

                    z = Variable(Tensor(np.random.normal(0, 1, (batch_size, *self.z_shape))))
                    X = self.netG(z)
                    d_validation, d_class = self.netD(X)

                    d_fake_loss = torch.mean(d_validation)

                    print(f"epoch {epoch} batch {index} d_gen_loss  : {d_fake_loss}")

                    d_fake_loss.backward()
                    self.optimizer_d.step()
                if not gif:
                    z = Variable(Tensor(np.random.normal(0, 1, (batch_size, *self.z_shape))))
                self.optimizer_g.zero_grad()

                target_classif_value = 1 / self.num_styles
                y_classif = np.zeros((batch_size)) + 1
                y_classif = Variable(torch.from_numpy(y_classif).cuda().type(torch.long))

                fake_images = self.netG(z)
                d_validation, d_class = self.netD(fake_images)

                # g_valid_loss = loss_d(d_validation, valid)
                g_valid_loss = -torch.mean(d_validation)
                if self.num_styles > 1:
                    g_label_loss = self.loss_d_label(d_class, y_classif)
                    print(f"epoch {epoch} batch {index} G_validity_loss : {g_valid_loss}, G_label_loss: {g_label_loss}")
                    g_total_loss = g_valid_loss + g_label_loss
                else:
                    print(f'epoch {epoch} batch {index} d_loss : {g_valid_loss}')
                    g_total_loss = g_valid_loss

                g_total_loss.backward()
                self.optimizer_g.step()

                print()
                if not gif:
                    index_diviser = 50
                else:
                    index_diviser = 10
                if index % index_diviser == 0:
                    fake_images = np.transpose(fake_images.cpu().detach().numpy(), (0, 2, 3, 1))
                    image = combine_images(fake_images)
                    image = image * 127.5 + 127.5
                    image = Image.fromarray(image.astype('uint8'))
                    if not gif:
                        img_name = 'epoch%d_%06d.png' % (epoch, index)
                    else:
                        img_name = '%06d.png' % (gif_i)
                    image.save(join(save_folder, img_name))
                    gif_i += 1

    def load_optimizer_(self, optimizator_params: dict):
        if optimizator_params['type'] == 'Adam':
            lr = optimizator_params['lr']
            b1 = optimizator_params['b1']
            b2 = optimizator_params['b2']
            optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(b1, b2))
            optimizer_d = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(b1, b2))
        else:
            optimizer_g = torch.optim.SGD(self.netG.parameters(), lr=0.0002, momentum=True)
            optimizer_d = torch.optim.SGD(self.netD.parameters(), lr=0.0002, momentum=True)

        return optimizer_g, optimizer_d

    def weights_init_normal_(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def model_saver_(self, epoch: int):
        save_path = join(getcwd(), 'saved_model')
        date_today = date.today()
        month, day = date_today.month, date_today.day

        model_folder = join(save_path, f'model_{day}.{month}_{epoch}')
        generator_folder = join(model_folder, 'generator.pth')
        discriminator_folder = join(model_folder, 'discriminator.pth')

        if exists(model_folder):
            shutil.rmtree(model_folder, ignore_errors=False)
        makedirs(model_folder)

        torch.save({'model_state_dict': self.netG.state_dict()}, generator_folder)
        torch.save({'model_state_dict': self.netD.state_dict()}, discriminator_folder)

    def model_load_(self, month_day: str, epoch_ini: int):
        pass


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train_another(data_path: str, gen_params: dict, discr_params: dict, optimizator_params: dict, img_shape: Tuple[int],
                  epochs=100, BATCH_SIZE=4, weights=False, month_day='', epoch_ini=0, missing_folders=[],
                  gif=False, folders=None):
    latent_dim = 100
    gif_i = 0
    if not gif:
        save_folder = 'generated'
    else:
        save_folder = 'generated_gif'
    data, num_styles, classes = get_data(data_path, missing_folders, folders)
    print(data[6])
    discr_params['num_classes'] = num_styles
    label_bin = LabelBinarizer()
    label_bin.fit(list(range(num_styles)))
    z_shape = gen_params['z_shape']

    epoch = ' ' + str(epoch_ini) + '_epoch'

    netG = Generator(img_shape, gen_params)
    netD = Discriminator(img_shape, discr_params)

    cuda = torch.cuda.is_available()
    if cuda:
        netG.cuda()
        netD.cuda()

    print(summary(netD, img_shape))
    print(summary(netG, z_shape))

    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    if optimizator_params['type'] == 'Adam':
        lr = optimizator_params['lr']
        b1 = optimizator_params['b1']
        b2 = optimizator_params['b2']
        optimizer_g = torch.optim.Adam(netG.parameters(), lr=lr, betas=(b1, b2))
        optimizer_d = torch.optim.Adam(netD.parameters(), lr=lr, betas=(b1, b2))
    else:
        optimizer_g = torch.optim.SGD(netG.parameters(), lr=0.0002, momentum=True)
        optimizer_d = torch.optim.SGD(netD.parameters(), lr=0.0002, momentum=True)

    loss_d = torch.nn.BCELoss()
    loss_d_label = torch.nn.CrossEntropyLoss()

    Tensor = torch.cuda.FloatTensor

    for epoch in range(epoch_ini, epochs):
        if epoch % 5 == 0:
            model_saver(netG, netD, epoch)
        for index in range(int(len(data) / BATCH_SIZE)):
            real_images, real_labels = get_images_classes(BATCH_SIZE, data, classes, img_shape,
                                                          batch_num=index, channel_first=True, )
            valid = Variable(Tensor(BATCH_SIZE, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(BATCH_SIZE, 1).fill_(0.0), requires_grad=False)

            real_images, real_labels = get_one_image(BATCH_SIZE, data, classes, img_shape, channel_first=True,
                                                     img_idx=6)
            real_images = Variable(torch.from_numpy(real_images).cuda().type(torch.float32))
            real_labels = Variable(torch.from_numpy(real_labels).cuda().type(torch.long))

            optimizer_d.zero_grad()
            if index % 2 == 0:
                # Train on real images

                X = real_images
                y_classif = real_labels

                d_validation, d_class = netD(X)

                # d_real_loss = loss_d(d_validation, valid)
                d_real_loss = -torch.mean(d_validation)

                d_loss = []
                d_loss.append(d_real_loss)
                if num_styles > 1:
                    d_label_real_loss = loss_d_label(d_class, y_classif)
                    d_loss.append(d_label_real_loss)
                    print("epoch %d batch %d D_validity_loss : %f, D_label_loss: %f" % (
                        epoch, index, d_loss[0], d_loss[1]))
                else:
                    print(f'epoch {epoch} batch {index} d_loss : {d_loss[0]}')

                d_total_real_loss = d_real_loss + d_label_real_loss
                d_total_real_loss.backward()
                optimizer_d.step()

            else:
                # Train on artificial images

                z = Variable(Tensor(np.random.normal(0, 1, (BATCH_SIZE, *z_shape))))
                X = netG(z)
                d_validation, d_class = netD(X)

                # d_fake_loss = loss_d(d_validation, fake)
                d_fake_loss = torch.mean(d_validation)

                print("epoch %d batch %d d_gen_loss  : %f" % (epoch, index, d_fake_loss))

                d_fake_loss.backward()
                optimizer_d.step()
            if not gif:
                z = Variable(Tensor(np.random.normal(0, 1, (BATCH_SIZE, *z_shape))))
            optimizer_g.zero_grad()

            target_classif_value = 1 / num_styles
            y_classif = np.zeros((BATCH_SIZE)) + 1
            y_classif = Variable(torch.from_numpy(y_classif).cuda().type(torch.long))

            fake_images = netG(z)
            d_validation, d_class = netD(fake_images)

            # g_valid_loss = loss_d(d_validation, valid)
            g_valid_loss = -torch.mean(d_validation)
            d_loss = []
            d_loss.append(g_valid_loss)
            if num_styles > 1:
                g_label_loss = loss_d_label(d_class, y_classif)
                d_loss.append(d_label_real_loss)
                print("epoch %d batch %d G_validity_loss : %f, G_label_loss: %f" % (epoch, index, d_loss[0], d_loss[1]))
                g_total_loss = g_valid_loss + g_label_loss
            else:
                print(f'epoch {epoch} batch {index} d_loss : {d_loss[0]}')
                g_total_loss = g_valid_loss

            g_total_loss.backward()
            optimizer_g.step()

            print()
            if not gif:
                index_diviser = 50
            else:
                index_diviser = 10
            if index % index_diviser == 0:
                fake_images = np.transpose(fake_images.cpu().detach().numpy(), (0, 2, 3, 1))
                image = combine_images(fake_images)
                image = image * 127.5 + 127.5
                image = Image.fromarray(image.astype('uint8'))
                if not gif:
                    img_name = 'epoch%d_%06d.png' % (epoch, index)
                else:
                    img_name = '%06d.png' % (gif_i)
                image.save(join(save_folder, img_name))
                gif_i += 1


if __name__ == '__main__':
    img_rows = 64
    img_cols = 64
    img_channels = 3
    img_shape = (img_channels, img_rows, img_cols)
    latent_dim = 100
    filter_size_g = (5, 5)
    filter_size_d = (5, 5)
    d_strides = (2, 2)

    color_mode = 'rgb'
    data_folder = 'data/images64'

    losses = ['binary_crossentropy', 'categorical_crossentropy']
    # losses = [wasserstein_loss, wasserstein_loss]

    gen_params = {
        'z_shape': (128, 2, 2),
        'nlastfilters': 64,
        'filter_size': filter_size_g
    }

    dis_params = {
        'init_filter_cnt': 16,
        'conv_cnt': 2,
        'drop_rate': 0.2,
        'filter_size': filter_size_d
    }
    optimizator_params = {
        'type': 'Adam',
        'lr': 0.0002,
        'b1': 0.5,
        'b2': 0.999
    }
    folders = ['abstraktnyy-ekspressionizm', 'zhivopis-tsvetovogo-polya', 'abstraktsionizm', 'minimalizm']

    trainer = TrainerGAN(data_folder, folders = folders)
    trainer.init_model(gen_params, dis_params, optimizator_params)
    trainer.train(100, 64)


    # train_another(data_folder, gen_params, dis_params, optimizator_params, img_shape, 100, 64, epoch_ini=0,
    #               month_day='',
    #               missing_folders=[], folders=['abstraktnyy-ekspressionizm', 'zhivopis-tsvetovogo-polya',
    #                                            'abstraktsionizm', 'minimalizm'], gif=False)
