# -*- coding: utf-8 -*-
"""EEGNet_fusion.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e81qW_BUO1QX1rWOw9-uKHRiy_5I9fF3
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import concatenate

"""
The author of this model is Karel Roots and was published along with the paper titled 
"Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification"
"""

def EEGNet(nb_classes, Chans = 64, Samples = 128,  
           conv_filt1 = 64, conv_filt2 = 128, conv_filt3 = 256, 
           separable_filt1 = 8, separable_filt2 = 16, separable_filt3 = 32,
           F1 = 4, F1_2 = 8, F1_3 = 16,
           F2 = 16, F2_2 = 16, F2_3 = 16, 
           D = 2, D2 = 2, D3 = 2,
           dropoutRate = 0.5, norm_rate = 0.25, dropoutType = 'Dropout'):
  
  if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
  elif dropoutType == 'Dropout':
      dropoutType = Dropout
  else:
      raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
  #-----------------------------------------------------------------------------
  # Branch 1
  input1 = Input(shape = (Chans, Samples, 1))
  block1 = Conv2D(F1, (1,conv_filt1), padding='same',
                  name='Conv2D_1',
                  input_shape = (Chans, Samples, 1),
                  use_bias = False)(input1)
  block1 = BatchNormalization()(block1)
  block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                            name='Depth_wise_Conv2D_1',
                            depth_multiplier=D,
                            depthwise_constraint=max_norm(1.))(block1)
  block1 = BatchNormalization()(block1)
  block1 = Activation('elu')(block1)
  block1 = AveragePooling2D((1,4))(block1)
  block1 = dropoutType(dropoutRate)(block1)

  block2 = SeparableConv2D(F2,(1,separable_filt1),
                            name='Separable_Conv2D_1',
                            use_bias = False, padding = 'same')(block1) 
  block2 = BatchNormalization()(block2)
  block2 = Activation('elu')(block2)
  block2 = AveragePooling2D((1,8))(block2)
  block2 = dropoutType(dropoutRate)(block2)
  block2 = Flatten()(block2) 
  #-----------------------------------------------------------------------------
  # Branch 2
  input2 = Input(shape = (Chans, Samples, 1))
  block3 = Conv2D(F1_2, (1,conv_filt2), padding='same',
                  name='Conv2D_2',
                  input_shape = (Chans, Samples, 1),
                  use_bias=False)(input2)
  block3 = BatchNormalization()(block3)
  block3 = DepthwiseConv2D((Chans, 1), use_bias=False,
                            name='Depth_wise_Conv2D_2',
                            depth_multiplier=D2,
                            depthwise_constraint=max_norm(1.))(block3)
  block3 = BatchNormalization()(block3)
  block3 = Activation('elu')(block3)
  block3 = AveragePooling2D((1,4))(block3)
  block3 = dropoutType(dropoutRate)(block3)

  block4 = SeparableConv2D(F2_2, (1,separable_filt2),
                            name='Separable_Conv2D_2',
                            use_bias=False, padding='same')(block3)  
  block4 = BatchNormalization()(block4)
  block4 = Activation('elu')(block4)
  block4 = AveragePooling2D((1,8))(block4)
  block4 = dropoutType(dropoutRate)(block4)
  block4 = Flatten()(block4)
  #-----------------------------------------------------------------------------
  # Branch 3
  input3 = Input(shape = (Chans, Samples, 1))
  block5 = Conv2D(F1_3, (1,conv_filt3), padding='same',
                  name='Conv2D_3',
                  input_shape = (Chans, Samples, 1),
                  use_bias=False)(input3)
  block5 = BatchNormalization()(block5)
  block5 = DepthwiseConv2D((Chans, 1), use_bias=False,
                            name='Depth_wise_Conv2D_3',
                            depth_multiplier=D3,
                            depthwise_constraint=max_norm(1.))(block5)
  block5 = BatchNormalization()(block5)
  block5 = Activation('elu')(block5)
  block5 = AveragePooling2D((1,4))(block5)
  block5 = dropoutType(dropoutRate)(block5)

  block6 = SeparableConv2D(F2_3, (1,separable_filt3),
                            name='Separable_Conv2D_3',
                            use_bias=False, padding='same')(block5)
  block6 = BatchNormalization()(block6)
  block6 = Activation('elu')(block6)
  block6 = AveragePooling2D((1,8))(block6)
  block6 = dropoutType(dropoutRate)(block6)
  block6 = Flatten()(block6)  
  #-----------------------------------------------------------------------------
  merge_one = concatenate([block2, block4])
  merge_two = concatenate([merge_one, block6])

  flatten = Flatten()(merge_two)

  dense = Dense(nb_classes, name='output',
                kernel_constraint=max_norm(norm_rate))(flatten)

  softmax = Activation('softmax', name='out_activation')(dense)

  return Model(inputs=[input1, input2, input3], outputs=softmax)
  #-----------------------------------------------------------------------------