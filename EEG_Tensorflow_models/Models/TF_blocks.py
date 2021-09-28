import tensorflow as tf


class TCN_residualblock(tf.keras.layers.Layer):
    def __init__(self, Ft,Kt,Dt,resconv=False,data_format='channels_first',dilation_rate=1,
                trainable=True, name=None, dtype=None, 
                 activity_regularizer=None, **kwargs):
        super(TCNresidualblock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.resconv = resconv   
        self.convolution0_1 = tf.keras.layers.Conv1D(Ft, kernel_size=Kt, strides=1, padding='causal', dilation_rate=dilation_rate,data_format=data_format)
        self.convolution0_2 = tf.keras.layers.Conv1D(Ft, kernel_size=Kt, strides=1, padding='causal', dilation_rate=dilation_rate*2,data_format=data_format)
        self.convolution0_3 = tf.keras.layers.Conv1D(Ft, kernel_size=Kt, strides=1, padding='causal', dilation_rate=dilation_rate*4,data_format=data_format)
        self.BatchNorm0 = tf.keras.layers.BatchNormalization()
        self.relu0 = tf.keras.layers.ELU()
        self.dropout0 = tf.keras.layers.Dropout(rate=Dt)

        self.convolution1_1 = tf.keras.layers.Conv1D(Ft, kernel_size=Kt, strides=1, padding='causal', dilation_rate=dilation_rate,data_format=data_format)
        self.convolution1_2 = tf.keras.layers.Conv1D(Ft, kernel_size=Kt, strides=1, padding='causal', dilation_rate=dilation_rate*2,data_format=data_format)
        self.convolution1_3 = tf.keras.layers.Conv1D(Ft, kernel_size=Kt, strides=1, padding='causal', dilation_rate=dilation_rate*4,data_format=data_format)
        self.BatchNorm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ELU()
        self.dropout1 = tf.keras.layers.Dropout(rate=Dt)
        self.residual = tf.keras.layers.Conv1D(1, kernel_size=1, padding='same')
    def call(self,inputs, training=True):
        x = self.convolution0_1(inputs)
        x = self.convolution0_2(x)
        x = self.convolution0_3(x)
        x = self.BatchNorm0(x)
        x = self.relu0(x)
        x = self.dropout0(x, training=training)
        x = self.convolution1_1(x)
        x = self.convolution1_2(x)
        x = self.convolution1_3(x)
        x = self.BatchNorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)
        if self.resconv:
            inputs = self.residual(inputs)
        return x + inputs
