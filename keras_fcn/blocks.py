import keras.backend as K
from keras.layers import (
    Dropout,
    Lambda
)
from keras.layers.convolutional import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    ZeroPadding2D
)
from keras.layers.merge import add
from keras_fcn.layers import CroppingLike2D

def vgg_conv(filters, convs, block_name='blockx'):
    """A VGG convolutional block for encoding.

    :param filters: Integer, number of filters per conv layer
    :param convs: Integer, number of conv layers in the block
    :param block_name: String, the name of the block, e.g., block1

    >>> from keras_fcn.blocks import vgg_conv
    >>> x = vgg_conv(filters=64, convs=2, block_name='block1')(x)

    """
    def f(x):
        for i in range(convs):
            if block_name == 'block1' and i == 0:
                pad = ZeroPadding2D(padding=(100, 100))(x)
                x = Conv2D(filters, (3, 3), activation='relu', padding='valid',
                           kernel_initializer='he_normal',
                           name='{}_conv{}'.format(block_name, int(i + 1)))(pad)
            else:
                x = Conv2D(filters, (3, 3), activation='relu', padding='same',
                           kernel_initializer='he_normal',
                           name='{}_conv{}'.format(block_name, int(i + 1)))(x)

        pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                         name='{}_pool'.format(block_name))(x)
        print(pool.get_shape())
        return pool
    return f


def vgg_fc(filters, block_name='block5'):
    """A fully convolutional block for encoding.

    :param filters: Integer, number of filters per fc layer

    >>> from keras_fcn.blocks import vgg_fc
    >>> x = vgg_fc(filters=4096)(x)

    """
    def f(x):
        fc6 = Conv2D(filters=4096, kernel_size=(7, 7),
                     activation='relu', padding='valid',
                     kernel_initializer='he_normal',
                     name='{}_fc6'.format(block_name))(x)
        drop6 = Dropout(0.5)(fc6)
        fc7 = Conv2D(filters=4096, kernel_size=(1, 1),
                     activation='relu', padding='valid',
                     kernel_initializer='he_normal',
                     name='{}_fc7'.format(block_name))(drop6)
        drop7 = Dropout(0.5)(fc7)
        return drop7
    return f


def vgg_deconv(classes, scale=1, kernel_size=(4, 4), strides=(2, 2),
               crop_offset='centered', block_name='featx'):
    """A VGG convolutional transpose block for decoding.

    :param classes: Integer, number of classes
    :param scale: Float, scale factor to the input feature, varing from 0 to 1
    :param kernel_size: Tuple, the kernel size for Conv2DTranspose layers
    :param strides: Tuple, the strides for Conv2DTranspose layers
    :param crop_offset: Tuple or "centered", the offset for cropping.
    The default is "centered", which crop the center of the feature map.

    >>> from keras_fcn.blocks import vgg_deconv
    >>> x = vgg_deconv(classes=21, scale=1e-2, block_name='feat2')(x)

    """
    def f(x, y):
        #def scaling(x, scale=1):
        #    return x * 0.5
        #s = Lambda(scaling, arguments={'scale': scale},
        #           name='scale_{}'.format(block_name))(x)
        score = Conv2D(filters=classes, kernel_size=(1, 1),
                   kernel_initializer='he_normal',
                   name='score_{}'.format(block_name))(x)
        if y is None:
            upscore = Conv2DTranspose(filters=classes, kernel_size=kernel_size,
                                strides=strides, padding='valid',
                                kernel_initializer='he_normal',
                                use_bias=False,
                                name='upscore_{}'.format(block_name))(score)
        else:
            crop = CroppingLike2D(target_shape=K.int_shape(y), offset=crop_offset,
                               name='crop_{}'.format(block_name))(score)
            merge = add([y, crop])
            upscore = Conv2DTranspose(filters=classes, kernel_size=kernel_size,
                                strides=strides, padding='valid',
                                kernel_initializer='he_normal',
                                use_bias=False,
                                name='upscore_{}'.format(block_name))(merge)
        return upscore
    return f


def vgg_score(crop_offset='centered'):
    """A helper block to crop the decoded feature.

    :param crop_offset: Tuple or "centered", the offset for cropping.
    The default is "centered", which crop the center of the feature map.

    >>> from keras_fcn.blocks import vgg_deconv
    >>> score = vgg_score(crop_offset='centered')(image, upscore)

    """
    def f(x, y):
        score = CroppingLike2D(target_shape=K.int_shape(x), offset=crop_offset, name='score')(y)
        return score
    return f
