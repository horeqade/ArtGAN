import tensorflow._api.v1.keras.backend as K
import shutil
import torch
from os.path import join, exists
from os import getcwd, makedirs
from datetime import date


def model_saver(generator: torch.nn.Module, discriminator: torch.nn.Module, epoch: int):
    save_path = join(getcwd(), 'saved_model')
    date_today = date.today()
    month, day = date_today.month, date_today.day

    model_folder = join(save_path, f'model_{day}.{month}_{epoch}')
    generator_folder = join(model_folder, 'generator.pth')
    discriminator_folder = join(model_folder, 'discriminator.pth')

    if exists(model_folder):
        shutil.rmtree(model_folder, ignore_errors=False)
    makedirs(model_folder)

    torch.save({'model_state_dict': generator.state_dict()}, generator_folder)
    torch.save({'model_state_dict': discriminator.state_dict()}, discriminator_folder)


def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


def get_gradients(model):
    """Return the gradient of every trainable weight in model

    Parameters
    -----------
    model : a keras model instance

    First, find all tensors which are trainable in the model. Surprisingly,
    `model.trainable_weights` will return tensors for which
    trainable=False has been set on their layer (last time I checked), hence the extra check.
    Next, get the gradients of the loss with respect to the weights.

    """
    for tensor in model.trainable_weights:
        print(tensor.name)
    for layer in model.get_layer('Generator').layers:
        print(layer.name)
    weights = [tensor for tensor in model.trainable_weights]  # if model.get_layer(tensor.name[:-2]).trainable]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)
