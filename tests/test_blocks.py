import keras.backend as K
from keras.layers import Input
from keras_fcn.blocks import vgg


def test_vgg():
    if K.image_data_format() == 'channels_first':
        x = Input(shape=(3, 224, 224))
        y1_shape = (None, 64, 111, 111)
        y2_shape = (None, 128, 55, 55)
    else:
        x = Input(shape=(224, 224, 3))
        y1_shape = (None, 111, 111, 64)
        y2_shape = (None, 55, 55, 128)

    block1 = vgg(filters=64, convs=2, block_name='block1')
    y = block1(x)
    assert K.int_shape(y) == y1_shape

    block2 = vgg(filters=128, convs=2, block_name='block2')
    y = block2(y)
    assert K.int_shape(y) == y2_shape
