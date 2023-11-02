"""
@author: Michele Gazzea
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Reshape, Dropout, add, multiply, concatenate, Input,Conv2DTranspose, BatchNormalization, Lambda, Activation, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from keras.regularizers import l2




def UNet_Attention(input_shape, n_filters = 16, n_classes = 1, dropout = False, task = "segmentation"):
    
    def conv_block(inputs, filters, kernel_size = 3, activation = 'elu'):
        x = Conv2D(filters, kernel_size, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if dropout:
            x = Dropout(0.1)(x)
        return x    
    
    def attention_block(gating, x, inter_channel):
        """  
        Attention mechanism.                 
        # Useful resources:
        # https://github.com/nabsabraham/focal-tversky-unet/blob/master/newmodels.py
        # https://medium.com/aiguys/attention-u-net-resunet-many-more-65709b90ac8b
        
        ------
        x: connection, it comes from the skip-connection --> better spatial rappresentation 
        g: gate signal, it comes from the deep part of the architecture --> better feature rappresentation 
        """
        g = Conv2D(inter_channel, (1, 1), padding='same')(gating)
        x1 = Conv2D(inter_channel, (1, 1), strides = (2,2), padding='same')(x)
        
        # The two vectors are summed element-wise. 
        # This process results in aligned weights becoming larger while unaligned weights becoming relatively smaller.
        psi = tf.keras.layers.Add()([x1, g])
        
        # The resultant vector goes through a eLU activation layer and a 1x1 convolution that collapses the dimensions to 1
        psi = tf.keras.layers.Activation('elu')(psi)
        psi = Conv2D(1, (1, 1), padding='same')(psi)
        
        # This vector goes through a sigmoid layer which scales the vector between the range [0,1],
        # producing the attention coefficients (weights), where coefficients closer to 1 indicate more relevant features.
        psi = tf.keras.layers.Activation('sigmoid')(psi)
        
        # The attention coefficients are upsampled to the original dimensions of the x vector
        upsample_psi = UpSampling2D(size=(2,2))(psi)
        x = tf.keras.layers.Multiply()([x, upsample_psi])
        
        # According to https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831
        # a final 1x1x1 convolution is used to consolidate the attention signal to original x dimensions
        result = Conv2D(x.shape[-1], kernel_size = 1, padding='same', kernel_initializer='HeNormal')(x)
        result = BatchNormalization()(result)
        return x


    inputs = Input(input_shape)
    
    # Contracting Path
    conv1 = conv_block(inputs, n_filters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, n_filters*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(pool2, n_filters*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    bottleneck = conv_block(pool3, n_filters*8)
    
    # Expansive Path
    a3 = attention_block(bottleneck, conv3, n_filters*4)
    u3 = UpSampling2D((2, 2))(bottleneck)
    concat3 = Concatenate()([u3, a3])
    upconv3 = conv_block(concat3, n_filters*4)
    
    a2 = attention_block(upconv3, conv2, n_filters*2)
    u2 = UpSampling2D((2, 2))(upconv3)
    concat2 = Concatenate()([u2, a2])
    upconv2 = conv_block(concat2, n_filters*2)
    
    a1 = attention_block(upconv2, conv1, n_filters)
    u1 = UpSampling2D((2, 2))(upconv2)
    concat3 = Concatenate()([u1, a1])
    upconv1 = conv_block(concat3, n_filters)
    
    if task == "segmentation":
        if n_classes == 1:
            output = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer = 'he_normal', padding='same')(upconv1)
        else:
            output = Conv2D(n_classes,1, activation='softmax', kernel_initializer = 'he_normal', padding='same')(upconv1)
            output = Reshape((input_shape[0]*input_shape[1], n_classes))(output)
        
    elif task == "regression":
        output = Conv2D(1, (1, 1), activation='linear', kernel_initializer = 'he_normal', padding='same')(upconv1)
        
    return Model(inputs = inputs, outputs = output)




def ResUNet_Attention(input_shape, n_filters = 16):
    """
    U-Net with Residual convolutions and Attention Gates
    """
    
    def ResidualBlock(inputs, n_filters, channel_axis = 3):
        """
        Residual block with 2x Residual conv -> BatchNorm -> ReLU
        """

        conv = Conv2D(n_filters,3, activation='elu', padding='same', kernel_initializer='HeNormal')(inputs)
        # conv = BatchNormalization(axis = channel_axis)(conv)

        conv = Conv2D(n_filters,3,  activation='elu', padding='same', kernel_initializer='HeNormal')(conv)
        # conv = BatchNormalization(axis = channel_axis)(conv)

        shortcut = Conv2D(n_filters,1, activation='elu', padding='same')(inputs)
        residual_path = add([shortcut,conv])
        out = residual_path
        # out = BatchNormalization(axis = channel_axis)(residual_path)
        return out
    
    
    def ResidualDecoderBlock(gate_layer, attention_layer, n_filters, channel_axis = 3):
        """
        Applies attention and upsamples gate-layer before applying residual block
        """
        
        def GatingBlock(inputs, output_dim, channel_axis = 3):
            """
            Resizes input channel dimensions to output_dim
            """
            conv = Conv2D(output_dim,1,padding='same',kernel_initializer='HeNormal')(inputs)    
            conv = BatchNormalization(axis = channel_axis)(conv)
            return Activation('elu')(conv)
        
        
        def repeat_channel(input, repeat_count):
            """
            repeat input feature channel repeat_count times
            """
            return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                                  arguments={'repnum': repeat_count})(input)
        
        
        def AttentionBlock(input, gate_in, out_shape):
            """
            Attention mechanism.
            """            
            # Useful resources:
            # https://github.com/nabsabraham/focal-tversky-unet/blob/master/newmodels.py
            # https://medium.com/aiguys/attention-u-net-resunet-many-more-65709b90ac8b
            
            shape_x = K.int_shape(input)

            theta_x = Conv2D(out_shape, (2, 2), strides=(2, 2), padding='same',kernel_initializer='HeNormal')(input) 

            phi_g = Conv2D(out_shape, (1, 1), padding='same',kernel_initializer='HeNormal')(gate_in)
            
            # The two vectors are summed element-wise. 
            # This process results in aligned weights becoming larger while unaligned weights becoming relatively smaller.
            concat_xg = add([phi_g, theta_x])
            
            # The resultant vector goes through a eLU activation layer and a 1x1 convolution that collapses the dimensions to 1
            act_xg = Activation('elu')(concat_xg)
            psi = Conv2D(1, (1, 1), padding='same',kernel_initializer='HeNormal')(act_xg)
            
            # This vector goes through a sigmoid layer which scales the vector between the range [0,1],
            # producing the attention coefficients (weights), where coefficients closer to 1 indicate more relevant features.
            sigmoid_xg = Activation('sigmoid')(psi)
            
            # The attention coefficients are upsampled to the original dimensions of the x vector
            upsample_psi = UpSampling2D(size=(2,2))(sigmoid_xg)
            upsample_psi = repeat_channel(upsample_psi, shape_x[3])

            y = multiply([upsample_psi, input])
            
            # According to https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831
            # a final 1x1x1 convolution is used to consolidate the attention signal to original x dimensions
            result = Conv2D(shape_x[3], (1, 1), padding='same',kernel_initializer='HeNormal')(y)
            result_bn = BatchNormalization()(result)
            return result_bn 
        
                
        gate = GatingBlock(gate_layer,n_filters)
        attention = AttentionBlock(attention_layer, gate, n_filters)
        up = UpSampling2D(size=(2,2), interpolation="bilinear")(gate_layer)
        up = concatenate([up,attention], axis=channel_axis)
        up_conv = ResidualBlock(up, n_filters)
        return up_conv
    
    
    
    """----- start ------"""
    ins = Input(input_shape)
    # Contracting path
    c1 = ResidualBlock(ins, n_filters)
    c1_pool = MaxPooling2D(pool_size = (2,2))(c1)
    c2 = ResidualBlock(c1_pool, n_filters*2)
    c2_pool = MaxPooling2D(pool_size = (2,2))(c2)
    c3 = ResidualBlock(c2_pool, n_filters*4)
    c3_pool = MaxPooling2D(pool_size = (2,2))(c3)

    #Bottleneck
    c4 = ResidualBlock(c3_pool, n_filters*8)

    # Expanding path
    u3 = ResidualDecoderBlock(c4,c3, n_filters*4)
    u2 = ResidualDecoderBlock(u3,c2, n_filters*2)
    u1 = ResidualDecoderBlock(u2,c1, n_filters)
    
    return Model(inputs=[ins], outputs=[u1])





def unet(input_shape, n_filters= 16, n_classes = 1, task = "segmentation"):
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
    
    
    if task == "segmentation":
        if n_classes == 1:
            output = Conv2D(1,1, activation='sigmoid', kernel_initializer = 'he_normal', padding='same')(u4)
        else:
            output = Conv2D(n_classes,1, activation='softmax', kernel_initializer = 'he_normal', padding='same')(u4)
            output = Reshape((input_shape[0]*input_shape[1], n_classes))(output)
            
    elif task == "regression":
        output = Conv2D(1, (1, 1), activation='linear', kernel_initializer = 'he_normal', padding='same')(u4)
        
    model = Model(inputs=[ins], outputs=[output])
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


