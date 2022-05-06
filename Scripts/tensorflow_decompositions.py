# Author: Matt Williams
# Version: 4/13/2022
# References: 
# - https://github.com/tarujg/keras-deep-compression/blob/master/tensor-decomposition/utils/tucker.py

import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import tensorflow as tf
from constants import *
#Includes pylance can't confirm
from tensorflow.keras.layers import Conv2D, ZeroPadding2D
from tensorflow.keras.models import Sequential


def _cp_decomposition_cnn_layer(layer, rank): 
    '''Given a Conv2D layer and an estimated ranked, perform CP decomposition on the given
        layer and return the new Conv2D layers. CURRENTLY NOT WORKING'''
    tl.set_backend("tensorflow")

    org_weights_bias_list = layer.get_weights()
    weights = tf.convert_to_tensor(org_weights_bias_list[0])

    layer_config = layer.get_config()

    print("----------------------------------------------------------")
    print("OG Layer Weights:",weights.shape)
    print("ranks", rank)

    weights, (horizontal, vertical, first, last) = \
        parafac(weights, rank=rank, init='random')

    print("New Weights:", weights.shape)
    print("----------------------------------------------------------")


    
    dilation_rate = layer_config['dilation_rate']
    strides = layer_config['strides']
    activation = layer_config['activation']

    depthwise_padding = 0
    if layer_config['padding'] == "same": 
        depthwise_padding = layer_config["kernel_size"][0] // 2


    pointwise_s_to_r_layer = Conv2D(filters = first.shape[1], \
            kernel_size = 1, strides = 1, padding ="valid", 
             dilation_rate=dilation_rate, use_bias = False)
    pointwise_s_to_r_layer.build(input_shape = [None, None, first.shape[0]])

    print("First:", first.shape)
    print(pointwise_s_to_r_layer.get_config())
    print("----------------------------------------------------------")

    padding_layer_1 = ZeroPadding2D(padding = (depthwise_padding, 0))

    depthwise_vertical_layer = Conv2D(filters = vertical.shape[1], \
            kernel_size=(vertical.shape[0], 1), strides = strides,
            padding="valid", dilation_rate = dilation_rate, 
            use_bias = False, groups = vertical.shape[1])
    depthwise_vertical_layer.build(input_shape = [None, None, first.shape[1]])

    print("Vertical:", vertical.shape)
    print(depthwise_vertical_layer. get_config())
    print("----------------------------------------------------------")

    padding_layer_2 = ZeroPadding2D(padding = (0, depthwise_padding))

    depthwise_horizontal_layer = Conv2D(filters = horizontal.shape[1], \
            kernel_size=(1, horizontal.shape[0]), strides = strides,
            padding="valid", dilation_rate = dilation_rate, 
            use_bias = False, groups = horizontal.shape[1])
    depthwise_horizontal_layer.build(input_shape = [None, None, vertical.shape[1]])


    print("Horizontal:", horizontal.shape)
    print(depthwise_horizontal_layer.get_config())
    print("----------------------------------------------------------")


    pointwise_r_to_t_layer = Conv2D(filters = last.shape[0], \
            kernel_size = 1, strides = strides, padding = "valid", 
            dilation_rate = dilation_rate, use_bias = True, activation = activation)
    pointwise_r_to_t_layer.build(input_shape = [None, None, horizontal.shape[1]])

    print("Last:", last.shape)
    print(pointwise_r_to_t_layer.get_config())
    print("----------------------------------------------------------")


    h_l_weights = tf.expand_dims(horizontal, axis = 0)
    h_l_weights = tf.expand_dims(h_l_weights, axis = 2)
    depthwise_horizontal_layer.set_weights([h_l_weights])


    v_l_weights = tf.expand_dims(vertical, axis = 1)
    v_l_weights = tf.expand_dims(v_l_weights, axis = len(v_l_weights.shape) - 1)
    depthwise_vertical_layer.set_weights([v_l_weights])


    s_to_r_weights = tf.expand_dims(first, axis = 0)
    s_to_r_weights = tf.expand_dims(s_to_r_weights, axis = 0)
    pointwise_s_to_r_layer.set_weights([s_to_r_weights])

    r_to_t_weights = tf.transpose(last, perm = [1,0])
    r_to_t_weights = tf.expand_dims(r_to_t_weights, axis = 0)
    r_to_t_weights = tf.expand_dims(r_to_t_weights, axis = 0)
    
    pointwise_r_to_t_layer.set_weights((r_to_t_weights, org_weights_bias_list[1]))

    return [pointwise_s_to_r_layer, padding_layer_1, depthwise_vertical_layer, 
            padding_layer_2, depthwise_vertical_layer, pointwise_r_to_t_layer]


def _tucker_decompositon_cnn_layer(layer, ranks):
    '''Given a Conv2D layer and a list of estimated ranks, perform Tucker decomposition on the given
        layer and return the new Conv2D layers. CURRENTLY BUGGED''' 
    tl.set_backend("tensorflow")


    org_weight_bias_list = layer.get_weights()
    weights = tf.convert_to_tensor(org_weight_bias_list[0])

    layer_config = layer.get_config()

    core, [first,last] = \
            partial_tucker(weights, modes = [2,3], rank = ranks, init = "random")

    print("----------------------------------------------------------")
    print(layer_config)
    print("OG Layer Weights:", weights.shape)
    print("Ranks:", ranks)

    dilation_rate = layer_config['dilation_rate']
    strides = layer_config['strides']
    kernel_size = layer_config['kernel_size']
    activation = layer_config['activation']

    padding = 0
    if layer_config['padding'] == 'same':
        padding = (kernel_size[0] // 2, kernel_size[1] // 2) 

    org_weight_bias_list = layer.get_weights()


    first_layer = Conv2D(filters = first.shape[1], \
            kernel_size = 1, strides = 1, padding = "valid", 
            dilation_rate = dilation_rate, use_bias = False )
    first_layer.build(input_shape = [None, None, first.shape[0]])

    print("----------------------------------------------------------")
    print("first:", first.shape)
    print(first_layer)
    print("Weight Shape:", first_layer.get_weights()[0].shape)

    padding_layer = ZeroPadding2D(padding = padding)

    core_layer = Conv2D(filters = core.shape[-1], \
            kernel_size = core.shape[0], strides = strides, 
            padding = "valid", dilation_rate = dilation_rate, use_bias = False)
    core_layer.build(input_shape = [None, None, core.shape[-2]])

    print("----------------------------------------------------------")
    print("core:", core.shape)
    print(core_layer)
    print("Weight Shape:", core_layer.get_weights()[0].shape)

    last_layer = Conv2D(filters = last.shape[0], \
            kernel_size = 1, strides = 1, padding = "valid",
            dilation_rate = dilation_rate, use_bias = True, activation = activation)
    last_layer.build([None, None, core.shape[-1]])
    print("----------------------------------------------------------")
    print("last:", last.shape)
    print(last_layer)
    print("Weight Shape:", last_layer.get_weights()[0].shape)
    print("----------------------------------------------------------")

    
    f_l_weights = tf.expand_dims(first, axis = 0)
    f_l_weights = tf.expand_dims(f_l_weights, axis = 0)
    first_layer.set_weights([f_l_weights])


    l_l_weights = tf.transpose(last, perm = [1,0])
    l_l_weights = tf.expand_dims(l_l_weights, axis = 0)
    l_l_weights = tf.expand_dims(l_l_weights, axis = 0)
    last_layer.set_weights((l_l_weights, org_weight_bias_list[1]))

    core_layer.set_weights([core])


    return [first_layer, padding_layer, core_layer, last_layer]


def decompose_cnn_layers(cnn_layers, decomposition = Decomposition.CP):
    """Given the CNN layers of a tensorflow model, perform the given decompositon on each of the 
    Conv2D layers except the first and return the new model. except the first.
    Not touching the first layer allows us to continue using the same Linear Layers."""
    
    count = 0
    decomposed_model = Sequential()
    found_first_cnn = False
    for i, layer in enumerate(cnn_layers.layers): 

        if type(layer) is Conv2D and not found_first_cnn: 
            decomposed_model.add(layer)
            found_first_cnn = True
            continue

        if type(layer) is not Conv2D:
            decomposed_model.add(layer)
            continue

        if decomposition == Decomposition.CP: 
            rank = estimate_cp_rank(layer, backend="tensorflow")
            decomposed_layers = _cp_decomposition_cnn_layer(layer, rank = rank)
            for decomposed_layer in decomposed_layers: 
                decomposed_model.add(decomposed_layer)
                count += 1
            count = 0 
        
        elif decomposition == Decomposition.Tucker: 
            ranks = estimate_tucker_ranks(layer, backend = "tensorflow")
            decomposed_layers = _tucker_decompositon_cnn_layer(layer, ranks = ranks)
            for decomposed_layer in decomposed_layers: 
                decomposed_model.add(decomposed_layer)

    return decomposed_model