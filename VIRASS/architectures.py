"""
@author: Michele Gazzea
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Reshape, Dropout, add, multiply, concatenate, Input,Conv2DTranspose, BatchNormalization, Lambda, Activation, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from keras.regularizers import l2





def ResUNet_Attention(input_shape,n_filters=64):
    """
    U-Net with Residual convolutions and Attention Gates
    """
    
    def ResidualBlock(inputs,n_filters=64,channel_axis = 3):
        """
        Residual block with 2x Residual conv -> BatchNorm -> ReLU
        """

        conv = Conv2D(n_filters,3,padding='same',kernel_initializer='HeNormal')(inputs)
        conv = BatchNormalization(axis = channel_axis)(conv)
        conv = Activation('relu')(conv)

        conv = Conv2D(n_filters,3,padding='same',kernel_initializer='HeNormal')(conv)
        conv = BatchNormalization(axis = channel_axis)(conv)
        conv = Activation('relu')(conv)

        #shortcut = Conv2D(n_filters,1,padding='same')(conv)  #Shouldn't it be (inputs) instead of (conv)?
        shortcut = Conv2D(n_filters,1,padding='same')(inputs)  # Modified by Michele
        residual_path = add([shortcut,conv])
        out = BatchNormalization(axis = channel_axis)(residual_path)
        return Activation('relu')(out)
    
    
    def ResidualDecoderBlock(gate_layer,attention_layer,n_filters = 64,channel_axis = 3):
        """
        Applies attention and upsamples gate-layer before applying residual block
        """
        
        def GatingBlock(inputs,output_dim,channel_axis = 3):
            """
            Resizes input channel dimensions to output_dim
            """

            conv = Conv2D(output_dim,1,padding='same',kernel_initializer='HeNormal')(inputs)    
            conv = BatchNormalization(axis = channel_axis)(conv)
            return Activation('relu')(conv)
        
        
        def repeat_channel(input, repeat_count):
            """
            repeat input feature channel repeat_count times
            """

            return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                                  arguments={'repnum': repeat_count})(input)
        
        def AttentionBlock(input,gate_in,out_shape):
            """
            Attention mechanism.
            Modified version of code from https://github.com/MoleImg/Attention_UNet/blob/master/AttResUNet.py
            """

            shape_x = K.int_shape(input)

            theta_x = Conv2D(out_shape, (2, 2), strides=(2, 2), padding='same',kernel_initializer='HeNormal')(input) 

            phi_g = Conv2D(out_shape, (1, 1), padding='same',kernel_initializer='HeNormal')(gate_in)
            upsample_g = Conv2DTranspose(out_shape, (3, 3),strides=(1,1),padding='same',kernel_initializer='HeNormal')(phi_g)
            concat_xg = add([upsample_g, theta_x])
            act_xg = Activation('relu')(concat_xg)
            psi = Conv2D(1, (1, 1), padding='same',kernel_initializer='HeNormal')(act_xg)
            sigmoid_xg = Activation('sigmoid')(psi)
            upsample_psi = UpSampling2D(size=(2,2))(sigmoid_xg)
            upsample_psi = repeat_channel(upsample_psi, shape_x[3])

            y = multiply([upsample_psi, input])

            result = Conv2D(shape_x[3], (1, 1), padding='same',kernel_initializer='HeNormal')(y)
            result_bn = BatchNormalization()(result)
            return result_bn 
        
        
        
        gate = GatingBlock(gate_layer,n_filters)
        attention = AttentionBlock(attention_layer,gate,n_filters)
        up = UpSampling2D(size=(2,2),interpolation="bilinear")(gate_layer)
        up = concatenate([up,attention],axis=channel_axis)
        up_conv = ResidualBlock(up,n_filters)
        return up_conv
    
    
    
    """----- start ------"""
    ins = Input(input_shape)
    # Down
    c1 = ResidualBlock(ins,n_filters)
    c1_pool = MaxPooling2D(pool_size = (2,2))(c1)
    c2 = ResidualBlock(c1_pool,n_filters*2)
    c2_pool = MaxPooling2D(pool_size = (2,2))(c2)
    c3 = ResidualBlock(c2_pool,n_filters*4)
    c3_pool = MaxPooling2D(pool_size = (2,2))(c3)
    c4 = ResidualBlock(c3_pool,n_filters*8)
    c4_pool = MaxPooling2D(pool_size = (2,2))(c4)

    #Bottleneck
    c5 = ResidualBlock(c4_pool,n_filters*16)

    # Up
    u4 = ResidualDecoderBlock(c5,c4,n_filters*8)
    u3 = ResidualDecoderBlock(u4,c3,n_filters*4)
    u2 = ResidualDecoderBlock(u3,c2,n_filters*2)
    u1 = ResidualDecoderBlock(u2,c1,n_filters)
    
    return Model(inputs=[ins], outputs=[u1])





def unet(input_shape, n_filters= 16, n_classes = 1):
    """
    Implementation of the U-Net described in the FFT U-Net paper:
    https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
    """
    
    # Blocks for regular U-Net
    def EncoderBlock(inputs, n_filters):
    
        conv = Conv2D(n_filters, 3, activation='elu', kernel_initializer = 'he_normal', padding='same')(inputs)
        conv = BatchNormalization()(conv)
        conv = Conv2D(n_filters, 3, activation='elu', kernel_initializer = 'he_normal', padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Dropout(0.1)(conv)
    
        next_layer = MaxPooling2D(pool_size = (2,2))(conv)       
        return next_layer, conv
    
    
    def DecoderBlock(prev_layer_input, skip_layer_input, n_filters=64):
        
        up = Conv2DTranspose(n_filters,(2,2),strides=(2,2),activation='elu',padding='same')(prev_layer_input)
        up = BatchNormalization()(up)
        merge = Concatenate(axis=3)([up, skip_layer_input])
    
        conv = Conv2D(n_filters, 3,activation='elu', kernel_initializer = 'he_normal', padding='same')(merge)
        conv = BatchNormalization()(conv)
    
        conv = Conv2D(n_filters,3, activation='elu', kernel_initializer = 'he_normal', padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Dropout(0.1)(conv)
        
        return conv



    # Contracting path
    ins = Input(input_shape)
    c1 = EncoderBlock(ins, n_filters)
    c2 = EncoderBlock(c1[0], n_filters*2)
    c3 = EncoderBlock(c2[0], n_filters*4)
    
    # Bottleneck
    bottleneck =  Conv2D(n_filters*8, 3, activation='elu', kernel_initializer = 'he_normal', padding='same')(c3[0])
    bottleneck = BatchNormalization()(bottleneck)
    bottleneck = Conv2D(n_filters*8, 3, activation='elu', kernel_initializer = 'he_normal', padding='same')(bottleneck)
    bottleneck = BatchNormalization()(bottleneck)
    
    # Expanding path
    u2 = DecoderBlock(bottleneck, c3[1], n_filters*2)
    u3 = DecoderBlock(u2, c2[1], n_filters*2)
    u4 = DecoderBlock(u3, c1[1], n_filters)
    
    
    # # Contracting path
    # ins = Input(input_shape)
    # c1 = EncoderBlock(ins, n_filters)
    # c2 = EncoderBlock(c1[0], n_filters*2)
    # c3 = EncoderBlock(c2[0], n_filters*4)
    # c4 = EncoderBlock(c3[0], n_filters*8)
    
    # # Bottleneck
    # bottleneck =  Conv2D(n_filters*16, 3, activation='elu', kernel_initializer = 'he_normal', padding='same')(c4[0])
    # bottleneck = BatchNormalization()(bottleneck)
    # bottleneck = Conv2D(n_filters*16, 3, activation='elu', kernel_initializer = 'he_normal', padding='same')(bottleneck)
    # bottleneck = BatchNormalization()(bottleneck)
    
    # # Expanding path
    # u1 = DecoderBlock(bottleneck, c4[1], n_filters*8)
    # u2 = DecoderBlock(u1, c3[1], n_filters*4)
    # u3 = DecoderBlock(u2, c2[1], n_filters*2)
    # u4 = DecoderBlock(u3, c1[1], n_filters)
 
    # conv9 = Conv2D(1,3, activation='sigmoid', padding='same')(u4)
    if n_classes == 1:
        conv9 = Conv2D(1,1, activation='sigmoid', kernel_initializer = 'he_normal', padding='same')(u4)
    else:
        conv9 = Conv2D(n_classes,1, activation='softmax', kernel_initializer = 'he_normal', padding='same')(u4)
        conv9 = Reshape((input_shape[0]*input_shape[1], n_classes))(conv9)
        
    model = Model(inputs=[ins], outputs=[conv9])
    return model


if __name__ == "main":
    model = unet(input_shape = (80,80, 3), n_filters = 16)
    tf.keras.utils.plot_model (model, to_file = 'unet_fixed.png', show_shapes = True, show_layer_names = True)




def autoencoder_model(input_shape = (None, None, 3)):
    
    input_img = Input(shape=input_shape)
    
    N_CHANNELS = input_shape[-1]
    
    x = Conv2D(8, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)     
    x = MaxPooling2D((2, 2), padding='same')(x)
   
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    encoded = MaxPooling2D((2, 2), padding='same')(x)
   
    x = Conv2D(8, (3, 3), padding='same')(encoded)    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(8, (3, 3), padding='same')(x)    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(N_CHANNELS, (3, 3), padding='same')(x)
    decoded = Activation('sigmoid')(x)
    
    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoded)
    
    return autoencoder, encoder


