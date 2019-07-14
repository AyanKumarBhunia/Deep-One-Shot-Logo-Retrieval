import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time

global batch_size  ########################################
batch_size = 32

def deconv2d(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             name='deconv2d', init_bias=0.):
  """Creates deconvolutional layers.

  Args:
    input_: 4D input tensor (batch size, height, width, channel).
    output_shape: Number of features in the output layer.
    k_h: The height of the convolutional kernel.
    k_w: The width of the convolutional kernel.
    d_h: The height stride of the convolutional kernel.
    d_w: The width stride of the convolutional kernel.
    stddev: The standard deviation for weights initializer.
    name: The name of the variable scope.
    init_bias: The initial bias for the layer.
  Returns:
    conv: The normalized tensor.
  """
  with tf.variable_scope(name):
    w = tf.get_variable('w',
                        [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                        initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
    biases = tf.get_variable('biases', [output_shape[-1]],
                             initializer=tf.constant_initializer(init_bias))
    deconv = tf.nn.bias_add(deconv, biases)
    deconv.shape.assert_is_compatible_with(output_shape)

    return deconv

def Network (Input1,Input2): #input1 : [Batch_size, 256, 256, 3], input2 : [Batch_size, 64, 64, 3]
    
    
    sess = tf.Session()
    
    
    with tf.name_scope ("Encoder"):
        
        
        
        conv1_1 = tf.layers.conv2d(Input1, filters = 64, 
                                   kernel_size = 3, strides = 1, padding='SAME', 
                                   activation = tf.nn.relu, name = 'conv1_1') #[32, 256, 256, 64]
        conv1_2 = tf.layers.conv2d(conv1_1, filters = 64, 
                                   kernel_size = 3, strides = 1, padding='SAME', 
                                   activation = tf.nn.relu, name = 'conv1_2') #[32, 256, 256, 64]
        pool1 = tf.layers.max_pooling2d(conv1_2, pool_size = 2,
                                   strides = 2, padding='SAME', name = 'pool1') #[32, 128, 128, 64]
        
        
        
        conv2_1 = tf.layers.conv2d(pool1, filters = 128, 
                                   kernel_size = 3, strides = 1, padding='SAME', 
                                   activation = tf.nn.relu, name = 'conv2_1')  #[32, 128, 128, 128]
        conv2_2 = tf.layers.conv2d(conv2_1, filters = 128, 
                                   kernel_size = 3, strides = 1, padding='SAME', 
                                   activation = tf.nn.relu, name = 'conv2_2')  #[32, 128, 128, 128]
        pool2 = tf.layers.max_pooling2d(conv2_2, pool_size = 2, 
                                   strides = 2, padding='SAME', name = 'pool2')  #[32, 64, 64, 128]
        
        
        
        conv3_1 = tf.layers.conv2d(pool2, filters = 256, kernel_size = 3, 
                                   strides = 1, padding='SAME', 
                                   activation = tf.nn.relu, name = 'conv3_1') #[32, 64, 64, 256]
        conv3_2 = tf.layers.conv2d(conv3_1, filters = 256, 
                                   kernel_size = 3, strides = 1, padding='SAME', 
                                   activation = tf.nn.relu, name = 'conv3_2')  #[32, 64, 64, 256]
        pool3 = tf.layers.max_pooling2d(conv3_2, pool_size = 2, 
                                   strides = 2, padding='SAME', name = 'pool3')  #[32, 32, 32, 256]
        
        
        
        conv4_1 = tf.layers.conv2d(pool3, filters = 512, kernel_size = 3, 
                                   strides = 1, padding='SAME', 
                                   activation = tf.nn.relu, name = 'conv4_1')  #[32, 32, 32, 512]
        conv4_2 = tf.layers.conv2d(conv4_1, filters = 512, kernel_size = 3, 
                                   strides = 1, padding='SAME', 
                                   activation = tf.nn.relu, name = 'conv4_2')  #[32, 32, 32, 512]
        pool4 = tf.layers.max_pooling2d(conv4_2, pool_size = 2, 
                                   strides = 2, padding='SAME', name = 'pool4')  #[32, 16, 16, 512]
        
        
        
        conv5_1 = tf.layers.conv2d(pool4, filters = 512, kernel_size = 3, 
                                   strides = 1, padding='SAME', 
                                   activation = tf.nn.relu, name = 'conv5_1')  #[32, 16, 16, 512]
        conv5_2 = tf.layers.conv2d(conv5_1, filters = 512, kernel_size = 3, 
                                   strides = 1, padding='SAME', 
                                   activation = tf.nn.relu, name = 'conv5_2')  #[32, 16, 16, 512]
        pool5 = tf.layers.max_pooling2d(conv5_2, pool_size = 2, 
                                   strides = 2, padding='SAME', name = 'pool5')  #[32, 8, 8, 512]
        
       
        
    with tf.name_scope("Conditional"):
        
        
        
        Bconv1_1 = tf.layers.conv2d(Input2, filters = 32, kernel_size = 3, 
                                    strides = 1, padding='SAME', 
                                    activation = tf.nn.relu, name = 'Bconv1_1') #[32, 64, 64, 32]
        Bconv1_2 = tf.layers.conv2d(Bconv1_1, filters = 32, kernel_size = 3, 
                                    strides = 1, padding='SAME', 
                                    activation = tf.nn.relu, name = 'Bconv1_2') #[32, 64, 64, 32]
        Bpool1 = tf.layers.max_pooling2d(Bconv1_2, pool_size = 2, 
                                    strides = 2, padding='SAME', name = 'Bpool1')  #[32, 32, 32, 32]
        
        
        
        Bconv2_1 = tf.layers.conv2d(Bpool1, filters = 64, kernel_size = 3, 
                                    strides = 1, padding='SAME', 
                                    activation = tf.nn.relu, name = 'Bconv2_1')  #[32 32, 32, 64]
        Bconv2_2 = tf.layers.conv2d(Bconv2_1, filters = 64, kernel_size = 3, 
                                    strides = 1, padding='SAME', 
                                    activation = tf.nn.relu, name = 'Bconv2_2')  #[32, 32, 32, 64]
        Bpool2 = tf.layers.max_pooling2d(Bconv2_2, pool_size = 2, 
                                    strides = 2, padding='SAME', name = 'Bpool2') #[32, 16, 16, 64]
        
        
        
        Bconv3_1 = tf.layers.conv2d(Bpool2, filters = 128, kernel_size = 3, 
                                    strides = 1, padding='SAME', 
                                    activation = tf.nn.relu, name = 'Bconv3_1')  #[32, 16, 16, 128]
        Bpool3 = tf.layers.max_pooling2d(Bconv3_1, pool_size = 2, 
                                    strides = 2, padding='SAME', name = 'Bpool3')  #[32, 8, 8, 128]
        
        
        
        Bconv4_1 = tf.layers.conv2d(Bpool3, filters = 256, kernel_size = 3, 
                                    strides = 1, padding='SAME', 
                                    activation = tf.nn.relu, name = 'Bconv4_1')  #[32, 8, 8, 256]
        Bpool4 = tf.layers.max_pooling2d(Bconv4_1, pool_size = 2, 
                                    strides = 2, padding='SAME', name = 'Bpool4')  #[32, 4, 4, 512]
        
        
        
        Bconv5_1 = tf.layers.conv2d(Bpool4, filters = 512, kernel_size = 3, 
                                    strides = 1, padding='SAME', 
                                    activation = tf.nn.relu, name = 'Bconv5_1')  #[32, 4, 4, 512]
        Bpool5 = tf.layers.max_pooling2d(Bconv5_1, pool_size = 2, 
                                    strides = 2, padding='SAME', name = 'Bpool5')  #[32, 2, 2, 512]
        
        
        Bconv6 = tf.layers.conv2d(Bpool5, filters = 512, kernel_size = 2, 
                                    strides = 1, activation = tf.nn.relu, name = 'Bconv6') #[32, 1, 1, 512]
        
        
  
        Btile1=tf.tile(Bconv6,[1,pool5.get_shape().as_list()[1],pool5.get_shape().as_list()[2],1])  #[32, 8, 8, 512]
    
        Btile2=tf.tile(Bconv6,[1,conv5_2.get_shape().as_list()[1],conv5_2.get_shape().as_list()[2],1]) #[32, 16, 16, 512]
         
        Btile3=tf.tile(Bconv6,[1,conv4_2.get_shape().as_list()[1],conv4_2.get_shape().as_list()[2],1])  #[32, 32, 32, 512]
        
        Btile4=tf.tile(Bconv6,[1,conv3_2.get_shape().as_list()[1],conv3_2.get_shape().as_list()[2],1])  #[32, 64, 64, 512]
        
        Btile5=tf.tile(Bconv6,[1,conv2_2.get_shape().as_list()[1],conv2_2.get_shape().as_list()[2],1])  #[32, 128, 128, 512]
        
        
       
        
    with tf.name_scope("Decoder"):
        
        
        #concatination
        efused_1= tf.concat([pool5,Btile1],-1, name='efused_1')  #[32, 8, 8, 1024]
        
        #3x3 convolution + 3x3 convolution + 2x2 upsampling
        Dconv1_1=tf.layers.conv2d(efused_1, filters=512, kernel_size=3, 
                                             strides=1, padding='SAME',
                                            activation=tf.nn.relu, name='Dconv1_1')  #[32, 8, 8, 512]
        
        Dconv1_2=tf.layers.conv2d(Dconv1_1, filters=512, kernel_size=3, 
                                             strides=1, padding='SAME',
                                            activation=tf.nn.relu, name='Dconv1_2')  #[32, 8, 8, 512]
        
        upsam1=tf.nn.relu(deconv2d(Dconv1_2, [batch_size,16,16,512], name='upsam1')) #[32, 16, 16, 512]]
     
    
    
        #concatination
        efused_2= tf.concat([conv5_2,Btile2],-1, name='efused_2')  #[32, 16, 16, 1024]
        
        #1x1 convolution
        fconv_1 = tf.layers.conv2d(efused_2, filters = 512, kernel_size = 1, 
                                            strides = 1, padding='SAME', 
                                            activation = tf.nn.relu, name = 'fconv_1')  #[32, 16, 16, 512]
        #concatination
        dfused_2= tf.concat([upsam1 ,fconv_1],-1, name='dfused_2')  #[32, 16, 16, 1024]
        
        #3x3 convolution + 3x3 convolution + 2x2 upsampling
        Dconv2_1=tf.layers.conv2d(dfused_2, filters=512, kernel_size=3, 
                                             strides=1, padding='SAME',
                                            activation=tf.nn.relu, name='Dconv2_1')  #[32, 16, 16, 512]
        
        Dconv2_2=tf.layers.conv2d(Dconv2_1, filters=512, kernel_size=3, 
                                             strides=1, padding='SAME',
                                            activation=tf.nn.relu, name='Dconv2_2')  #[32, 16, 16, 512]
        
        upsam2 =tf.nn.relu(deconv2d(Dconv2_2, [batch_size,32,32,512], name='upsam2')) #[32, 32, 32, 512]

        
        
        
        #concatination
        efused_3= tf.concat([conv4_2,Btile3],-1, name='efused_3')  #[32, 32, 32, 1024]
        
        #1x1 convolution
        fconv_2 = tf.layers.conv2d(efused_3, filters = 256, kernel_size = 1, 
                                           strides = 1, padding='SAME', 
                                           activation = tf.nn.relu, name = 'fconv_2')  #[32, 32, 32, 256]
        #concatination
        dfused_3= tf.concat([upsam2 ,fconv_2],-1, name='dfused_3')  #[32, 32, 32, 768]
        
        #3x3 convolution + 3x3 convolution + 2x2 upsampling
        Dconv3_1=tf.layers.conv2d(dfused_3, filters=256, kernel_size=3, 
                                             strides=1, padding='SAME',
                                            activation=tf.nn.relu, name='Dconv3_1')  #[32, 32, 32, 256]
        
        Dconv3_2=tf.layers.conv2d(Dconv3_1, filters=256, kernel_size=3, 
                                             strides=1, padding='SAME',
                                            activation=tf.nn.relu, name='Dconv3_2')  #[32, 32, 32, 256]
        
        upsam3 = tf.nn.relu(deconv2d(Dconv3_2, [batch_size,64,64,256], name='upsam3'))  #[32, 64, 64, 256]

        
    
    
    
        #concatination
        efused_4= tf.concat([conv3_2,Btile4],-1, name='efused_4')  #[32, 64, 64, 768]
        
        #1x1 convolution
        fconv_3 = tf.layers.conv2d(efused_4, filters = 128, kernel_size = 1, 
                                           strides = 1, padding='SAME', 
                                           activation = tf.nn.relu, name = 'fconv_3')  #[32, 64, 64, 128]
        #concatination
        dfused_4= tf.concat([upsam3 ,fconv_3],-1, name='dfused_4')  #[32, 64, 64, 384]
        
        #3x3 convolution + 3x3 convolution + 2x2 upsampling
        Dconv4_1=tf.layers.conv2d(dfused_4, filters=128, kernel_size=3, 
                                             strides=1, padding='SAME',
                                            activation=tf.nn.relu, name='Dconv4_1')  #[32, 64, 64, 128]
        
        Dconv4_2=tf.layers.conv2d(Dconv4_1, filters=128, kernel_size=3, 
                                             strides=1, padding='SAME',
                                            activation=tf.nn.relu, name='Dconv4_2')  #[32, 64, 64, 128]
        
        upsam4 = tf.nn.relu(deconv2d(Dconv4_2, [batch_size,128,128,128], name='upsam4'))  #[32, 128, 128, 128]
 





        #concatination
        efused_5= tf.concat([conv2_2,Btile5],-1, name='efused_5')  #[32, 128, 128, 640]
        
        #1x1 convolution
        fconv_4 = tf.layers.conv2d(efused_5, filters = 64, kernel_size = 1, 
                                           strides = 1, padding='SAME', 
                                           activation = tf.nn.relu, name = 'fconv_4')  #[32, 128, 128, 64]
        #concatination
        dfused_5= tf.concat([upsam4 ,fconv_4],-1, name='dfused_5')  #[32, 128, 128, 192]
        
        
        #3x3 convolution + 3x3 convolution + 2x2 upsampling
        Dconv5_1=tf.layers.conv2d(dfused_5, filters=64, kernel_size=3, 
                                            strides=1, padding='SAME',
                                           activation=tf.nn.relu, name='Dconv5_1')  #[32, 128, 128, 64]
        
        Dconv5_2=tf.layers.conv2d(Dconv5_1, filters=64, kernel_size=3, 
                                            strides=1, padding='SAME',
                                           activation=tf.nn.relu, name='Dconv5_2')  #[32, 128, 128, 64]
        
        upsam5 = tf.nn.relu(deconv2d(Dconv5_2, [batch_size,256,256,64], name='upsam5'))  #[32, 256, 256, 64]
        
        
        output=tf.layers.conv2d(upsam5, filters=1, kernel_size=3, 
                                           strides=1,  padding='SAME', name='fconv_5')  #[32, 256, 256, 1]
        
        
    
    output = tf.identity(output, 'Output')
        
        
    trainwriter = tf.summary.FileWriter('./log_dir14/', sess.graph)
        

tf.reset_default_graph()



Input1=tf.placeholder(dtype = tf.float32, shape = [batch_size, 256, 256, 3], name = 'Target')
Input2=tf.placeholder(dtype = tf.float32, shape = [batch_size, 64, 64, 3], name = 'Query')


Output = Network(Input1,Input2)



