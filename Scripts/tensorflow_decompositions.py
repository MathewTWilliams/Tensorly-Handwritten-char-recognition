# Author: Matt Williams
# Version: 4/13/2022

import tensorly as tl
from tensorly.decomposition import parafac, tucker
import tensorflow as tf

#Includes pylance can't confirm
from tensorflow.keras.layers import Conv2D, ZeroPadding2D


def cp_decomposition_cnn_layer(layer, rank): 

    tl.set_backend("tensorflow")

    org_weight_bias_tuple = layer.get_weights()

    last, first, vertical, horizontal = \
        parafac(org_weight_bias_tuple[0], rank=rank, init='svd')

    layer_config = layer.get_config()
    dilation_rate = layer_config['dilation_rate']
    strides = layer_config['strides']
    depthwise_padding = 0
    if layer_config['padding'] == "same": 
        depthwise_padding = layer_config["kernel_size"][0] // 2


    pointwise_s_to_r_layer = Conv2D(filters = first.shape[1], \
            kernel_size = 1, strides = 1, padding ="valid", 
             dilation_rate=dilation_rate, use_bias = False)

    padding_layer_1 = ZeroPadding2D(padding = (depthwise_padding, 0))

    depthwise_vertical_layer = Conv2D(filters = vertical.shape[1], \
            kernel_size=(vertical.shape[0], 1), strides = strides,
            padding="valid", dilation_rate = dilation_rate, 
            use_bias = False, groups = vertical.shape[1])

    padding_layer_2 = ZeroPadding2D(padding = (0, depthwise_padding))

    depthwise_horizontal_layer = Conv2D(filters = horizontal.shape[1], \
            kernel_size=(1, horizontal.shape[0]), strides = strides,
            padding="valid", dilation_rate = dilation_rate, 
            use_bias = False, groups = horizontal.shape[1])

    pointwise_r_to_t_layer = Conv2D(filters = last.shape[0], \
            kernel_size = 1, strides = strides, padding = "valid", 
            dilation_rate = dilation_rate, use_bias = True)

    #TODO perm might not be correct
    h_l_weights = tf.transpose(horizontal, perm = [1,0,2,3]) 
    h_l_weights = tf.expand_dims(h_l_weights, axis = 1)
    h_l_weights = tf.expand_dims(h_l_weights, axis = 1)
    depthwise_horizontal_layer.set_weights(h_l_weights)

    v_l_weights = tf.transpose(vertical, perm = [1,0,2,3]) 
    v_l_weights = tf.expand_dims(v_l_weights, axis = 1)
    v_l_weights = tf.expand_dims(v_l_weights, axis = len(v_l_weights.shape) - 1)
    depthwise_vertical_layer.set_weights(v_l_weights)


    s_to_r_weights = tf.transpose(first, perm = [1,0,2,3])
    s_to_r_weights = tf.expand_dims(s_to_r_weights, axis = len(s_to_r_weights.shape) - 1)
    s_to_r_weights = tf.expand_dims(s_to_r_weights, axis = len(s_to_r_weights.shape) - 1)
    pointwise_s_to_r_layer.set_weights(s_to_r_weights)

    r_to_t_weights = tf.expand_dims(last, axis = len(last.shape) - 1)
    r_to_t_weights = tf.expand_dims(last, axis = len(r_to_t_weights.shape) - 1)

    pointwise_r_to_t_layer.set_weights(tuple(r_to_t_weights, org_weight_bias_tuple[1]))

    return [pointwise_s_to_r_layer, padding_layer_1, depthwise_vertical_layer, 
            padding_layer_2, depthwise_vertical_layer, pointwise_r_to_t_layer]