from keras.models import Model, load_model
import tensorflow as tf
from keras.layers import Input, BatchNormalization, Activation, Add, UpSampling2D
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras import optimizers
from keras.layers import BatchNormalization
from tensorflow.keras.metrics import MeanIoU
import keras


"""Res-Unet""" 
def myResUNet(input_shape=(None, None, 8)):
        
    def bn_act(x, act=True):
        x = BatchNormalization()(x)
        if act == True:
            x = Activation("relu")(x)
            # x = Dropout(0.2)(x)
        return x

    def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1, act=True):
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = bn_act(conv)
        return conv
    
    def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1)        
        shortcut = Conv2D(filters, kernel_size=(1, 1), padding="same", strides=strides)(x)
        
        output = Add()([conv, shortcut])
        return output
    
    def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
        
        shortcut = Conv2D(filters, kernel_size=(1, 1), padding="same", strides=strides)(x)
        
        output = Add()([shortcut, res])
        return output
    
    def upsample_concat_block(x, xskip):
        u = UpSampling2D((2, 2))(x)
        c = concatenate([u, xskip], axis=3)
        return c


    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((input_shape))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    d4 = Dropout(0.1)(d4)
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    model = keras.models.Model(inputs, outputs)
    
    model.summary()
    return model