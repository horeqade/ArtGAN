from math import log2
from typing import Tuple

from tensorflow._api.v1.keras.layers import (Dense, Activation, Reshape, Flatten, Dropout, Input, LeakyReLU, Conv2D,
                                     Conv2DTranspose, BatchNormalization)
from tensorflow._api.v1.keras.models import Sequential, Model


def generator_containing_discriminator(g: Model, d: Model, d_label: Model, latent_dim: int = 100):
    noise = Input(shape=(latent_dim,))
    img = g(noise)
    d.trainable = False
    d_label.trainable = False
    valid, target_label = d(img), d_label(img)

    return Model(noise, [valid, target_label])


def build_dcnn_generator(img_shape: Tuple[int], dense_shape: Tuple[int, int, int], init_filters_cnt: int,
                         img_channels: int = 3, input_dim: int = 100, filter_size: Tuple[int] = (3, 3)):
    out_side = img_shape[0]
    layers_cnt = int(log2(out_side) - log2(dense_shape[0]))
    dense_units = dense_shape[0] * dense_shape[1] * dense_shape[2]

    model = Sequential(name='Generator')
    model.add(Dense(dense_units, input_dim=input_dim))
    # model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((dense_shape)))

    filters_denum = 1
    for i in range(layers_cnt):
        model.add(Conv2DTranspose(int(init_filters_cnt / filters_denum), filter_size, strides=(2,2), padding='same'))
        # model.add(BatchNormalization())
        model.add(LeakyReLU())
        filters_denum *= 2

    model.add(Conv2DTranspose(img_channels, filter_size, strides=(1, 1), padding='same'))
    model.add(Activation("tanh"))

    model.summary()

    return model


def build_dcnn_discriminator_classes(num_classes: int, img_shape: Tuple[int], init_filter_cnt: int = 64,
                                     conv_cnt: int = 6, filter_size: Tuple[int] = (3, 3), drop_rate: float = 0.5):
    model = Sequential(name='FeatureExtractor')

    model.add(Conv2D(init_filter_cnt, kernel_size=filter_size, strides=(2, 2), input_shape=img_shape, padding="same"))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    # model.add(Dropout(drop_rate))

    filters_cnt = init_filter_cnt * 2
    for i in range(conv_cnt - 1):
        model.add(Conv2D(filters_cnt, kernel_size=filter_size, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        # model.add(Dropout(drop_rate))
        filters_cnt *= 2

    model.add(Flatten())
    model.summary()

    img = Input(shape=img_shape)

    features = model(img)

    validity = Dense(1)(features)
    valid = Activation('sigmoid')(validity)

    label1 = Dense(int(filters_cnt/2))(features)
    bn1 = BatchNormalization()(label1)
    lrelu1 = LeakyReLU()(bn1)
    label2 = Dense(int(filters_cnt/4))(lrelu1)
    bn2 = BatchNormalization()(label2)
    lrelu2 = LeakyReLU()(bn2)
    label3 = Dense(num_classes)(lrelu2)
    label = Activation('softmax')(label3)

    return Model(img, valid, name='Discriminator'), Model(img, label, name='ClassDiscriminator')
    # return Model(img, [valid, label])
