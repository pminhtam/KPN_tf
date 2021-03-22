import tensorflow as tf
from model.basic_layer import Basic

def kernel_predic(input_tensor,core):
    b,h,w,c = input_tensor.shape
    # print(input_tensor)
    # input_tensor_pad = tf.zeros((None,h+4,w+4,c))
    input_tensor = tf.pad(input_tensor,tf.constant([[0,0],[2, 2,], [2, 2],[0,0]]))
    # input_tensor_pad[:,2:h+2,2:w+2,:] = input_tensor
    # input_tensor = input_tensor_pad
    # print(input_tensor.shape)
    # print(core[:,:,:,0].shape)
    # print(input_tensor[:,0:h,0:w,:].shape)
    img1 = input_tensor[:,0:h,0:w,:]*core[:,:,:,0:1]
    img2 = input_tensor[:,1:h+1,0:w,:]*core[:,:,:,1:2]
    img3 = input_tensor[:,2:h+2,0:w,:]*core[:,:,:,2:3]
    img4 = input_tensor[:,0:h,1:w+1,:]*core[:,:,:,3:4]
    img5 = input_tensor[:,1:h+1,1:w+1,:]*core[:,:,:,4:5]
    img6 = input_tensor[:,2:h+2,1:w+1,:]*core[:,:,:,5:6]
    img7 = input_tensor[:,0:h,2:w+2,:]*core[:,:,:,6:7]
    img8 = input_tensor[:,1:h+1,2:w+2,:]*core[:,:,:,7:8]
    img9 = input_tensor[:,2:h+2,2:w+2,:]*core[:,:,:,8:]
    # print(img1.shape)
    img = img1+img2+img3+img4+img5+img6+img7+img8+img9
    # img2 = input_tensor[:,1:h+1,1:w+1,:]*core[1]
    return img
def kpn(input_tensor):
    out_ch = 3**2
    conv1 = Basic(input_tensor,32)
    conv2 = tf.nn.max_pool2d(conv1,2,2,padding='SAME')
    conv2 = Basic(conv2,64)
    conv3 = tf.nn.max_pool2d(conv2,2,2,padding='SAME')
    conv3 = Basic(conv3,128)
    conv4 = tf.nn.max_pool2d(conv3,2,2,padding='SAME')
    conv4 = Basic(conv4,128)
    conv4_up = tf.keras.layers.UpSampling2D(
        size=(2, 2), data_format=None, interpolation='nearest')(conv4)
    conv5 = Basic(tf.keras.layers.Concatenate(axis=-1)([conv3,conv4_up]),128)
    conv5_up = tf.keras.layers.UpSampling2D(
        size=(2, 2), data_format=None, interpolation='nearest')(conv5)
    conv6 = Basic(tf.keras.layers.Concatenate(axis=-1)([conv2,conv5_up]),64)
    conv6_up = tf.keras.layers.UpSampling2D(
        size=(2, 2), data_format=None, interpolation='nearest')(conv6)

    conv7 =  Basic(tf.keras.layers.Concatenate(axis=-1)([conv1,conv6_up]),out_ch)
    feature_map = tf.nn.softmax(conv7)
    img_re = kernel_predic(input_tensor,feature_map)
    # img_re = feature_map
    return img_re

def kpn_old(input_tensor):
    out_ch = 3**2
    feature_map = Basic(input_tensor,64)
    # feature_map = Basic(feature_map,64)
    feature_map = Basic(feature_map,128)
    feature_map = Basic(feature_map,128)
    feature_map = Basic(feature_map,out_ch)
    feature_map = tf.nn.softmax(feature_map)

    img_re = kernel_predic(input_tensor,feature_map)
    # img_re = feature_map
    return img_re
input_tensor = tf.keras.Input(shape=[256, 256, 3])
core = tf.keras.Input(shape=[256, 256, 9])
output_tensor = kpn(input_tensor)
# output_tensor = kpn_old(input_tensor)
model = tf.keras.Model(input_tensor,output_tensor)
model.summary()
# print(kernel_predic(input_tensor,core))