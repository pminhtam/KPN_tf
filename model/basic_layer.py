import tensorflow as tf


def spatial_attention_block(input_tensor):
    """Spatial Attention Block"""
    average_pooling = tf.reduce_max(input_tensor, axis=-1)
    # print(average_pooling)
    average_pooling = average_pooling[:,:,:,tf.newaxis]
    # average_pooling = tf.expand_dims(average_pooling, axis=-1)
    # print(average_pooling)
    max_pooling = tf.reduce_mean(input_tensor, axis=-1)
    max_pooling = max_pooling[:,:,:,tf.newaxis]
    # max_pooling = tf.expand_dims(max_pooling, axis=-1)
    concatenated = tf.keras.layers.Concatenate(axis=-1)([average_pooling, max_pooling])
    feature_map = tf.keras.layers.Conv2D(1, kernel_size=(1, 1))(concatenated)
    feature_map = tf.nn.sigmoid(feature_map)
    return input_tensor * feature_map


def channel_attention_block(input_tensor):
    """Channel Attention Block"""
    channels = list(input_tensor.shape)[-1]
    average_pooling = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    feature_descriptor = tf.reshape(average_pooling, shape=(-1, 1, 1, channels))
    feature_activations = tf.keras.layers.ReLU()(
        tf.keras.layers.Conv2D(
            filters=channels // 8, kernel_size=(1, 1)
        )(feature_descriptor)
    )
    feature_activations = tf.nn.sigmoid(
        tf.keras.layers.Conv2D(
            filters=channels, kernel_size=(1, 1)
        )(feature_activations)
    )
    return input_tensor * feature_activations

def Basic(input_tensor,out_ch):
    input_shape = input_tensor.shape[1:]
    feature_map = input_tensor
    feature_map = tf.keras.layers.Conv2D(filters=out_ch,kernel_size=(3,3),padding='same',input_shape=input_shape)(feature_map)
    # feature_map = tf.keras.layers.BatchNormalization(axis=-1)(feature_map)
    feature_map = tf.keras.layers.ReLU()(feature_map)
    feature_map = tf.keras.layers.Conv2D(filters=out_ch,kernel_size=(3,3),padding='same',input_shape=input_shape)(feature_map)
    # feature_map = tf.keras.layers.BatchNormalization(axis=-1)(feature_map)
    feature_map = tf.keras.layers.ReLU()(feature_map)
    feature_map = tf.keras.layers.Conv2D(filters=out_ch,kernel_size=(3,3),padding='same',input_shape=input_shape)(feature_map)
    feature_map = tf.keras.layers.BatchNormalization(axis=-1)(feature_map)
    feature_map = tf.keras.layers.ReLU()(feature_map)
    feature_map = channel_attention_block(feature_map)
    feature_map = spatial_attention_block(feature_map)
    return feature_map

