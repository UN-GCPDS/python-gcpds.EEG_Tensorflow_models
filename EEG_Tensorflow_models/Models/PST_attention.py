from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow import Variable
import tensorflow as tf


class Channel_attention(Model):
    def __init__(self,num_ch):
        super(Channel_attention,self).__init__(name='')
        self.conv1      = Conv2D(filters=8,kernel_size=(1,1),kernel_constraint = max_norm(2., axis=(0,1,2)))
        self.conv2      = Conv2D(filters=8,kernel_size=(1,1),kernel_constraint = max_norm(2., axis=(0,1,2)))
        self.reshape    = Reshape((num_ch,-1))
        self.gamma      = Variable(initial_value=0,dtype='float32',trainable=True)

    def call(self,x):
        proj_query      = self.reshape(self.conv1(x))
        proj_key        = self.reshape(self.conv2(x))

        energy          = tf.matmul(proj_query,proj_key,transpose_b=True)

        max_H           = tf.math.reduce_max(energy,axis=2,keepdims=True)
        min_H           = tf.math.reduce_min(energy,axis=2,keepdims=True)

        tmp_b           = (energy-min_H)
        tmp_c           = (max_H-min_H)+1e-8
        energy          = tmp_b/tmp_c
        attention       = tf.nn.softmax(energy)

        out = self.gamma * tf.expand_dims(tf.matmul(attention,tf.squeeze(x,axis=-1),transpose_a=True),axis=3) + x

        return out

class Time_attention(tf.keras.Model):
    def __init__(self,num_t):
        super(Time_attention,self).__init__(name='')
        self.conv1      = Conv2D(filters=8,kernel_size=(1,1),kernel_constraint = max_norm(2., axis=(0,1,2)))
        self.conv2      = Conv2D(filters=8,kernel_size=(1,1),kernel_constraint = max_norm(2., axis=(0,1,2)))
        self.reshape    = Reshape((num_t,-1))
        self.permute    = Permute((2,1,3))
        self.gamma      = Variable(initial_value=0,dtype='float32',trainable=True)

    def call(self,x):
        b,ch,t,d        = x.shape
        proj_query      = self.reshape(self.permute(self.conv1(x)))
        proj_key        = self.reshape(self.permute(self.conv2(x)))

        energy          = tf.matmul(proj_query,proj_key,transpose_b=True)

        max_H           = tf.math.reduce_max(energy,axis=2,keepdims=True)
        min_H           = tf.math.reduce_min(energy,axis=2,keepdims=True)

        tmp_b           = (energy-min_H)
        tmp_c           = (max_H-min_H)+1e-8
        energy          = tmp_b/tmp_c
        attention       = tf.nn.softmax(energy)

        out = self.gamma * tf.expand_dims(tf.matmul(tf.squeeze(x,axis=-1),attention,transpose_b=True),axis=3) + x

        return out

class Attention(tf.keras.Model):
    def __init__(self,num_ch,num_t):
        super(Attention,self).__init__()
        self.CA = Channel_attention(num_ch=num_ch)
        self.TA = Time_attention(num_t=num_t)

    def call(self,x,training=None):
        out1 = self.CA(x)
        out2 = self.TA(x)

        out = tf.concat([x,out1,out2],axis=-1)
        
        return out



def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 



def PST_attention(nb_classes=4, Chans = 22, Samples = 128, dropoutRate = 0.5,last_layer = 'Conv'):

    bias_spatial = False
    pool         = (1,75)
    strid        = (1,15)
    filters      = (1,25)


    input_main   = Input((Chans, Samples, 1))
    block1       = Attention(num_ch=Chans,num_t=Samples)(input_main)
    block1       = Conv2D(40, filters, 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = Conv2D(40, (Chans, 1), use_bias=bias_spatial, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=pool, strides=strid)(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)

    if last_layer=='Conv':
        ConvC    = Conv2D(nb_classes, (1, block1.shape[2]),kernel_constraint = max_norm(0.5, axis=(0,1,2)),name='ouput')(block1)
        flat     = Flatten(name='F_1')(ConvC)
        softmax  = Activation('softmax',name='A_out')(flat)

    elif last_layer=='Dense':
        flatten  = Flatten(name='F_1')(block1)
        dense    = Dense(nb_classes, kernel_constraint = max_norm(0.5),name='output')(flatten)
        softmax  = Activation('softmax',name='A_out')(dense)

    
    return Model(inputs=input_main, outputs=softmax)