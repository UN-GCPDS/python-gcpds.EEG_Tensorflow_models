from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D,Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.losses import CategoricalCrossentropy



def DMTL_BCI(nb_classes, Chans = 22, Samples = 250, dropoutRate = 0.5):

    filters      = (1,40)
    strid        = (1,15)
    pool         = (1,75)
    bias_spatial = True

    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(40, filters, strides=(1,2),
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=bias_spatial, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    Act1         = Activation('elu')(block1)
    block1       = AveragePooling2D(pool_size=pool, strides=strid)(Act1)
    block1       = Dropout(dropoutRate)(block1)

    ConvC        = Conv2D(nb_classes, (1, block1.shape[2]),kernel_constraint = max_norm(0.5, axis=(0,1,2)),name='ouput')(block1)
    flat         = Flatten(name='F_1')(ConvC)
    softmax      = Activation('softmax',name='A_out')(flat)

    block2       = Resizing(Act1.shape[1], Act1.shape[2])(block1)
    block2       = Conv2DTranspose(40, (Chans, 1), use_bias=bias_spatial, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block2       = Conv2DTranspose(40, filters,strides=(1,2),
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    
    return Model(inputs=input_main, outputs=[block2,softmax])
