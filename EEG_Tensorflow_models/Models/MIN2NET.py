from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, AveragePooling2D,Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten,Reshape
from tensorflow.keras.constraints import max_norm

def MIN2NET(nb_classes=2, Chans = 20, Samples = 400, latent_dim = 20):

    filter_1         = Chans
    subsampling_size= 100
    pool_size_1     = (1,Samples//subsampling_size)
    filter_2         = 10
    pool_size_2     = (1,4)
    #latent_dim     = 8 or C or 64 or 256 
    flatten_size     = Samples//pool_size_1[1]//pool_size_2[1]

    encoder_input   = Input((1, Samples, Chans))

    block1          = Conv2D(filter_1, (1,64),padding='same', activation='elu',
                             kernel_constraint = max_norm(2., axis=(0,1,2)))(encoder_input)                                                                  
    block1          = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(block1)
    block1          = AveragePooling2D(pool_size= pool_size_1)(block1)  


    block2          = Conv2D(filter_2, (1, 32), activation='elu', padding="same", 
                                kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2          = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(block2)
    block2          = AveragePooling2D(pool_size= pool_size_2)(block2)

    block2          = Flatten()(block2)
    encoder_output  = Dense(latent_dim, kernel_constraint=max_norm(0.5))(block2)
    encoder         = Model(inputs=encoder_input, outputs=encoder_output, name='encoder')


    decoder_input   = Input(shape=(latent_dim,), name='decoder_input')

    block3          = Dense(1* flatten_size* filter_2, activation='elu', 
                               kernel_constraint=max_norm(0.5))(decoder_input)
    block3          = Reshape((1, flatten_size, filter_2))(block3)
    block3          = Conv2DTranspose(filters= filter_2, kernel_size=(1, 64), 
                                         activation='elu', padding='same', strides= pool_size_2, 
                                         kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    decoder_output  = Conv2DTranspose(filters= filter_1, kernel_size=(1, 32), 
                                         activation='elu', padding='same', strides= pool_size_1, 
                                         kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    decoder         = Model(inputs=decoder_input, outputs=decoder_output, name='decoder')


    x_encod         = encoder(encoder_input)
    x_decod         = decoder(x_encod)
    z               = Dense(nb_classes, activation='softmax', kernel_constraint=max_norm(0.5), 
                            name='classifier')(x_encod)

    return Model(inputs=encoder_input, outputs=[x_decod, x_encod, z], name='MIN2Net')