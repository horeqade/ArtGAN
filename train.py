from datetime import date
import os

from PIL import Image
from PIL import ImageFile
from tensorflow._api.v1.keras.optimizers import Adam, SGD
from tensorflow._api.v1.keras.utils import to_categorical
import tensorflow._api.v1.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from ArtistAI.model import build_dcnn_generator, build_dcnn_discriminator_classes, generator_containing_discriminator
from ArtistAI.data_processor import get_data, get_images_classes, combine_images
from ArtistAI.util import get_layer_output_grad, get_gradients

ImageFile.LOAD_TRUNCATED_IMAGES = True


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)




def model_saver(generator, discriminator, d_label, epoch: int):
    date_today = date.today()

    month, day = date_today.month, date_today.day

    # Генерируем описание модели в формате json
    d_json = discriminator.to_json()
    # Записываем модель в файл
    json_file = open(os.getcwd() + "/jsons/%d.%d dis_model.json" % (day, month), "w")
    json_file.write(d_json)
    json_file.close()

    # Генерируем описание модели в формате json
    d_l_json = d_label.to_json()
    # Записываем модель в файл
    json_file = open(os.getcwd() + "/jsons/%d.%d dis_label_model.json" % (day, month), "w")
    json_file.write(d_l_json)
    json_file.close()

    # Генерируем описание модели в формате json
    gen_json = generator.to_json()
    # Записываем модель в файл
    json_file = open(os.getcwd() + "/jsons/%d.%d gen_model.json" % (day, month), "w")
    json_file.write(gen_json)
    json_file.close()

    discriminator.save_weights(os.getcwd() + '/weights/%d.%d %d_epoch dis_weights.h5' % (day, month, epoch))
    d_label.save_weights(os.getcwd() + '/weights/%d.%d %d_epoch dis_label_weights.h5' % (day, month, epoch))
    generator.save_weights(os.getcwd() + '/weights/%d.%d %d_epoch gen_weights.h5' % (day, month, epoch))


def train_another(data_path: str, gen_params: dict, discr_params: dict, epochs=100, BATCH_SIZE=4, weights=False,
                  month_day='', epoch_ini=0, missing_folders=[],
                  gif=False, folders=None):
    gif_i = 0
    if not gif:
        save_folder = 'generated'
    else:
        save_folder = 'generated_gif'
    data, num_styles, classes = get_data(data_path, missing_folders, folders)
    discr_params['num_classes'] = num_styles

    epoch = ' ' + str(epoch_ini) + '_epoch'

    generator = build_dcnn_generator(**gen_params)
    discriminator, d_label = build_dcnn_discriminator_classes(**discr_params)

    if month_day != '':
        generator.load_weights(os.getcwd() + '/weights/' + month_day + epoch + ' gen_weights.h5', by_name=True)
        discriminator.load_weights(os.getcwd() + '/weights/' + month_day + epoch + ' dis_weights.h5', by_name=True)
        d_label.load_weights(os.getcwd() + '/weights/' + month_day + epoch + ' dis_label_weights.h5', by_name=True)

    dcgan = generator_containing_discriminator(generator, discriminator, d_label)

    discriminator.compile(loss=losses[0], optimizer=d_optim)
    d_label.compile(loss=losses[1], optimizer=d_optim)
    generator.compile(loss=losses[0], optimizer=g_optim)
    dcgan.compile(loss=losses[0], optimizer=g_optim)

    noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    real = np.ones(BATCH_SIZE)
    fake = np.zeros(BATCH_SIZE)
    for epoch in range(epoch_ini, epochs):
        if epoch % 5 == 0:
            model_saver(generator, discriminator, d_label, epoch)
        for index in range(int(len(data) / BATCH_SIZE)):

            if not gif:
                noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
            generated_images = generator.predict(noise)

            real_images, real_labels = get_images_classes(BATCH_SIZE, data, classes, gen_params['img_shape'],
                                                          batch_num=index)

            if index % 2 == 0:
                X = real_images
                # y_classif = real_labels - 0.2 + np.random.rand(BATCH_SIZE) * 0.2
                y_classif = to_categorical(np.zeros(BATCH_SIZE) + real_labels, num_styles)
                #y = 0.7 + np.random.rand(BATCH_SIZE) * 0.3
                # y = np.ones(BATCH_SIZE)

                d_loss = []
                d_loss.append(discriminator.train_on_batch(X, real))
                if num_styles > 1:
                    d_loss.append(d_label.train_on_batch(X, y_classif))
                    print("epoch %d batch %d d_loss : %f, label_loss: %f" % (epoch, index, d_loss[0], d_loss[1]))
                else:
                    print(f'epoch {epoch} batch {index} d_loss : {d_loss[0]}')

                # X = generated_images
                # y = np.random.rand(BATCH_SIZE) * 0.2
                # # y = np.zeros(BATCH_SIZE)
                # d_loss = discriminator.train_on_batch(X, fake)
                #
                # print("epoch %d batch %d d_gen_loss  : %f" % (epoch, index, d_loss))
            else:
                X = generated_images
                # y = np.random.rand(BATCH_SIZE) * 0.3
                # y = np.zeros(BATCH_SIZE)
                d_loss = discriminator.train_on_batch(X, fake)

                print("epoch %d batch %d d_gen_loss  : %f" % (epoch, index, d_loss))

                # X = real_images
                # # real_labels = real_labels - 0.1 + np.random.rand(BATCH_SIZE) * 0.2
                # y_classif = to_categorical(np.zeros(BATCH_SIZE) + real_labels, num_styles)
                # y = 0.8 + np.random.rand(BATCH_SIZE) * 0.2
                # # y = np.ones(BATCH_SIZE)
                #
                # d_loss = []
                # d_loss.append(discriminator.train_on_batch(X, y))
                # # discriminator.trainable = False
                # if num_styles > 1:
                #     d_loss.append(d_label.train_on_batch(X, y_classif))
                #     print("epoch %d batch %d d_loss : %f, label_loss: %f" % (epoch, index, d_loss[0], d_loss[1]))
                # else:
                #     print(f'epoch {epoch} batch {index} d_loss : {d_loss[0]}')

            if not gif:
                noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))

            # y_classif = keras.utils.to_categorical(np.zeros(BATCH_SIZE) + 1/num_styles, num_styles)
            target_classif_value = 1 / num_styles
            y_classif = np.zeros((BATCH_SIZE, num_styles))
            y_classif[:, 1] = 1

            # y = 0.7 + np.random.rand(BATCH_SIZE) * 0.3

            g_loss = dcgan.train_on_batch(noise, [real, y_classif])
            print("epoch %d batch %d g_loss     : %f" % (epoch, index, g_loss[0]))

            # print(dcgan.optimizer)
            # print(get_layer_output_grad(dcgan, noise, [y, y_classif], layer=0))
            # gradients = get_gradients(dcgan)
            # input_tensors = [
            #     dcgan.inputs[0],
            #     dcgan.sample_weights[0],
            #     dcgan.targets,
            #     K.learning_phase()
            # ]
            # get_gradients_f = K.function(inputs=input_tensors, outputs=gradients)
            # inputs = [
            #     noise,
            #     [1],
            #     [y, y_classif],
            #     0
            # ]
            # print(get_gradients(dcgan), get_gradients_f(inputs))

            if not gif:
                index_diviser = 50
            else:
                index_diviser = 10
            if index % index_diviser == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                image = Image.fromarray(image.astype('uint8'))
                if not gif:
                    img_name = 'epoch%d_%06d.png' % (epoch, index)
                else:
                    img_name = '%06d.png' % (gif_i)
                image.save(os.path.join(save_folder, img_name))
                # cv2.imwrite(
                #    os.getcwd() + '\\%s\\%s' % (save_folder, img_name), image)
                gif_i += 1
                # image = Image.fromarray(combine_images(real_images).astype('uint8'))
                # image = image*127.5+127.5
                # image.save(os.path.join(save_folder, img_name))
                # cv2.imwrite(
                #    os.getcwd() + '\\%s\\%s' % (save_folder, img_name), image)


if __name__ == '__main__':
    img_rows = 64
    img_cols = 64
    img_channels = 3
    img_shape = (img_rows, img_cols, img_channels)
    latent_dim = 100
    filter_size_g = (5, 5)
    filter_size_d = (5, 5)
    d_strides = (2, 2)

    color_mode = 'rgb'
    data_folder = 'data/images64'

    losses = ['binary_crossentropy', 'categorical_crossentropy']
    # losses = [wasserstein_loss, wasserstein_loss]

    gen_params = {
        'img_shape': img_shape,
        'dense_shape': (4, 4, 1024),
        'init_filters_cnt': 512,
        'img_channels': 3,
        'filter_size': filter_size_g
    }

    dis_params = {
        'img_shape': img_shape,
        'init_filter_cnt': 128,
        'conv_cnt': 3,
        'drop_rate': 0.2,
        'filter_size': filter_size_d
    }

    # g_optim = SGD(lr=0.0002, momentum=0.9, nesterov=True)
    # d_optim = SGD(lr=0.0002, momentum=0.9, nesterov=True)
    # g_optim = Adam(0.0002, beta_2 = 0.9)
    # d_optim = Adam(0.0002, beta_2 = 0.9)
    g_optim = Adam(0.0005, beta_2=0.5)
    d_optim = Adam(0.0005, beta_2=0.5)

train_another(data_folder, gen_params, dis_params, 100, 64, epoch_ini=0, month_day='', missing_folders=[],
              folders=['realizm', 'zhivopis-tsvetovogo-polya'], gif=False)
