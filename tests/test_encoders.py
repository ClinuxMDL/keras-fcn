import numpy as np
import keras.backend as K
from keras.layers import Input
from keras_fcn.encoders import (
    VGG16,
    VGG19)

from keras.utils.test_utils import keras_test


@keras_test
def test_vgg16():
    if K.image_data_format() == 'channels_last':
        x = Input(shape=(500, 500, 3))
        pool3_shape = (None, 88, 88, 256)
        pool4_shape = (None, 44, 44, 512)
        drop7_shape = (None, 16, 16, 4096)
    else:
        x = Input(shape=(3, 500, 500))
        pool3_shape = (None, 256, 88, 88)
        pool4_shape = (None, 512, 44, 44)
        drop7_shape = (None, 4096, 16, 16)

    encoder = VGG16(x, weights='imagenet', trainable=False)
    feat_pyramid = encoder.outputs
    for feat in feat_pyramid:
        if feat.name.startswith('block3_pool'):
            assert K.int_shape(feat) == pool3_shape
        elif feat.name.startswith('block4_pool'):
            assert K.int_shape(feat) == pool4_shape
        elif feat.name.startswith('dropout'):
            assert K.int_shape(feat) == drop7_shape
        elif feat.name.startswith('block5_pool'):
            print(feat.name)
            assert False
    assert len(feat_pyramid) == 5

    for layer in encoder.layers:
        if layer.name == 'block1_conv1':
            assert layer.trainable is False
            weights = K.eval(layer.weights[0])
            assert np.allclose(weights[0, 0, 0, 0], 0.429471)

    encoder_from_scratch = VGG16(x, weights=None, trainable=True)
    for layer in encoder_from_scratch.layers:
        if layer.name == 'block1_conv1':
            assert layer.trainable is True
            weights = K.eval(layer.weights[0])
            assert weights[0, 0, 0, 0] < 0.1


@keras_test
def test_vgg19():
    if K.image_data_format() == 'channels_last':
        x = Input(shape=(500, 500, 3))
        pool3_shape = (None, 88, 88, 256)
        pool4_shape = (None, 44, 44, 512)
        drop7_shape = (None, 16, 16, 4096)
    else:
        x = Input(shape=(3, 500, 500))
        pool3_shape = (None, 256, 88, 88)
        pool4_shape = (None, 512, 44, 44)
        drop7_shape = (None, 4096, 16, 16)

    encoder = VGG19(x, weights='imagenet', trainable=False)
    feat_pyramid = encoder.outputs
    for feat in feat_pyramid:
        if feat.name.startswith('block3_pool'):
            assert K.int_shape(feat) == pool3_shape
        elif feat.name.startswith('block4_pool'):
            assert K.int_shape(feat) == pool4_shape
        elif feat.name.startswith('dropout'):
            assert K.int_shape(feat) == drop7_shape
        elif feat.name.startswith('block5_pool'):
            print(feat.name)
            assert False
    assert len(feat_pyramid) == 5

    for layer in encoder.layers:
        if layer.name == 'block1_conv1':
            assert layer.trainable is False
            weights = K.eval(layer.weights[0])
            assert np.allclose(weights[0, 0, 0, 0], 0.429471)

    encoder_from_scratch = VGG19(x, weights=None, trainable=True)
    for layer in encoder_from_scratch.layers:
        if layer.name == 'block1_conv1':
            assert layer.trainable is True
            weights = K.eval(layer.weights[0])
            assert weights[0, 0, 0, 0] < 0.1
