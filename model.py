import tensorflow as tf
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, UpSampling2D
from tensorflow.keras.applications import VGG19
print('TensorFlow Version:', tf.__version__)
assert float(tf.__version__[:3]) > 2.0, 'Please Update TensorFlow To 2.0'
import numpy as np

### SRGAN has three neural networks, a generator, a discriminator, and a pre-trained VGG19 network on the Imagenet dataset
keras = tf.keras
# For Debuging
tf.config.experimental_run_functions_eagerly(True)


    ########################                  ########################
    ########################     GENERATOR    ########################
    ########################                  ########################

def _pixel_with_lambda(scale):
  return lambda x: tf.nn.depth_to_space(x, scale)

def _residual_block(x_input):
  x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x_input)
  x = BatchNormalization()(x)
  x = PReLU(shared_axes=[1, 2])(x)

  x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
  x = BatchNormalization()(x)
  x = Add()([x_input, x])

  return x


def model_generator(input_shape=(64, 64, 3)):

    x_input = Input(shape=input_shape, name='Generator_Input')

    conv_1 = Conv2D(64, kernel_size=9, strides=1, padding='same', name='G_Conv_1')(x_input)
    conv_1 = PReLU(shared_axes=[1, 2])(conv_1)

    # This Copy Need For Layers After Residual Block
    x_copy = conv_1

    ### RESIDUAL BLOCKS * 5
    conv_2 = _residual_block(conv_1)
    conv_3 = _residual_block(conv_2)
    conv_4 = _residual_block(conv_3)
    conv_5 = _residual_block(conv_4)
    conv_6 = _residual_block(conv_5)

    conv_7 = Conv2D(64, kernel_size=3, strides=1, padding='same', name='G_Conv_7')(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Add()([x_copy, conv_7])

    # Last Two Layers k=3, n=256, s=1
    conv_8 = Conv2D(256, kernel_size=3, strides=1, padding='same', name='G_Conv_8')(conv_7)
    conv_8 = keras.layers.Lambda(_pixel_with_lambda(scale=2))(conv_8)
    conv_8 = PReLU(shared_axes=[1, 2])(conv_8)

    conv_9 = Conv2D(256, kernel_size=3, strides=1, padding='same', name='G_Conv_9')(conv_8)
    conv_9 = keras.layers.Lambda(_pixel_with_lambda(scale=2))(conv_9)
    conv_9 = PReLU(shared_axes=[1, 2])(conv_9)

    conv_10 = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh', name='G_Conv_10')(conv_9)


    return keras.Model(x_input, conv_10, name='Generator')


    ########################                  ########################
    ########################  DISCRIMINATOR   ########################
    ########################                  ########################



def _d_conv2d_layers(input_layer, filters, stride, name, batch=True):
  x = Conv2D(filters, kernel_size=3, strides=stride, padding='same', name=name)(input_layer)
  if batch:
    x = BatchNormalization()(x)
  return LeakyReLU()(x)


def model_discriminator(input_shape = (256, 256, 3), filters=64):
  x_input = Input(shape=input_shape, name='D_Input')
  #x = keras.layers.Lambda(normalize_m11)(x_input)

  x = _d_conv2d_layers(x_input, filters, stride=1, name='D_Conv2D_1', batch=False)
  x = _d_conv2d_layers(x, filters, stride=2, name='D_Conv2D_2')

  x = _d_conv2d_layers(x, filters*2, stride=1, name='D_Conv2D_3')
  x = _d_conv2d_layers(x, filters*2, stride=2, name='D_Conv2D_4')

  x = _d_conv2d_layers(x, filters*4, stride=1, name='D_Conv2D_5')
  x = _d_conv2d_layers(x, filters*4, stride=2, name='D_Conv2D_6')

  x = _d_conv2d_layers(x, filters*8, stride=1, name='D_Conv2D_7')
  x = _d_conv2d_layers(x, filters*8, stride=2, name='D_Conv2D_8')

  x = Flatten()(x)
  x = Dense(1024)(x)
  x = LeakyReLU()(x)
  x = Dense(1, activation='sigmoid')(x)

  return keras.Model(x_input, x, name='Discriminator')


    ########################                              ########################
    ########################  PRE-TRAIN (VGG19) Network   ########################
    ########################                              ########################


def model_vgg19(output_layer):
    """
        We will use the pre-trained VGG19 network.
        The purpose of the VGG19 network is to extract feature maps of the generated and the real images.
    """
    ##11 or 16
    vgg = VGG19(
        input_shape=[None, None, 3],
        include_top=False)


    return keras.Model(vgg.input,
                       vgg.layers[output_layer].output)
