from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K




def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5, version = '2017',last_layer='Conv'):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """

    if version=='2017':
        bias_spatial = False
        pool = (1,3)
        strid = (1,3)
        filters = (1,10)
    elif version=='2018':
        bias_spatial = True
        pool = (1,2)
        strid = (1,2)
        filters = (1,5)


    # start the modelDeepConvNet(
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(25, filters, 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)),use_bias=bias_spatial)(block1) # Bias False en repo
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=pool, strides=strid)(block1)
    block1       = Dropout(dropoutRate)(block1) 
  
    block2       = Conv2D(50, filters,
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=pool, strides=strid)(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, filters,
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=pool, strides=strid)(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, filters,
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=pool, strides=strid)(block4)
    block4       = Dropout(dropoutRate)(block4)# igual a repo
    

    if version=='2017'or last_layer=='Conv':
        ConvC        = Conv2D(nb_classes, (1, block4.shape[2]),kernel_constraint = max_norm(0.5, axis=(0,1,2)))(block4)
        flat          = Flatten()(ConvC)
        softmax      = Activation('softmax')(flat)

    elif version=='2018' or last_layer=='Dense':
        flatten      = Flatten()(block4)
        dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
        softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)
