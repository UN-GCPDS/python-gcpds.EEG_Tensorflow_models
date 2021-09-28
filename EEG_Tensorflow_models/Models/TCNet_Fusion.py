from tensorflow.keras.layers import Reshape, Dropout, Activation, BatchNormalization, Concatenate
from tensorflow.keras.layers import Input, Flatten, Dense, Softmax
from tensorflow.keras.layers import SeparableConv2D, AveragePooling2D, DepthwiseConv2D, Conv2D
from tensorflow.keras.constraints import MaxNorm
from .TF_blocks import TCN_residualblock

def TCNet_Fusion(nb_classes, Chans = 64, Samples = 128, Ke_1 = 32, 
                 pool=(1,8), stride = (1,1), F2=12, Pe=0.3, Ke_2=16,Ft=12,
                 Kt=4,Pt=0.3,max_norm_1=1,max_norm_2=0.25):

  input_layer = Input(shape = [1,Chans,Samples])
  C1 = Conv2D(filters=F1, kernel_size=(1,Ke_1),padding='same',data_format='channels_first')(input_layer)
  BN1 = BatchNormalization()(C1)
  C2 =  DepthwiseConv2D(kernel_size=(Chans,1), strides=stride, padding='valid', depth_multiplier=D,data_format='channels_first',
                                        kernel_constraint=MaxNorm(max_value=max_norm_1))(BN1)
  BN2 = BatchNormalization()(C2)
  A1 = Activation('elu')(BN2)
  P1 = AveragePooling2D(pool_size=pool,data_format='channels_first')(A1)
  D1 = Dropout(rate=Pe)(P1)
  C3 = SeparableConv2D(F2, kernel_size=(1,Ke_2), strides=stride, padding='same',data_format='channels_first')(D1)
  BN3 = BatchNormalization()(C3)
  A2 = Activation('elu')(BN3)
  P2 = AveragePooling2D(pool_size=pool,data_format='channels_first')(A2)
  D2 = Dropout(rate=Pe)(P2)
  Rs = Reshape((F2,-1))(D2)
  # Two custom residual blocks
  TCN1 =  TCN_residualblock(Ft,Kt,Pt,resconv=False,data_format='channels_first',dilation_rate=1)(Rs)    
  TCN2 =  TCN_residualblock(Ft,Kt,Pt,resconv=False,data_format='channels_first',dilation_rate=1)(TCN1) 
  CON1 = Concatenate(axis=1)([Rs, TCN2])
  FC1 = Flatten()(D1)
  FC2 = Flatten()(CON1)
  CON2 = Concatenate(axis=1)([FC1,FC2])
  Dense = Dense(units=nb_classes,kernel_constraint=MaxNorm(max_value=max_norm_2))(CON2)
  Softmax = Softmax()(Dense)
  return tf.keras.Model(inputs=input_layer,outputs=Softmax)
