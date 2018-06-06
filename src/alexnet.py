# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
from keras.engine import Layer
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Lambda
from keras.layers import concatenate, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """

    def f(X):
        b, ch, r, c = X.shape.as_list()
        half = n // 2
        # Add some zero channels
        extra_channels = K.spatial_2d_padding(K.square(X), ((half, half), (0, 0)), data_format='channels_last')
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:, i:i + ch, :, :]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        div = X.shape.as_list()[axis] // ratio_split

        if axis == 0:
            output = X[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = X[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = X[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = X[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError('This axis is not possible')

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


def convolution2Dgroup(n_group, nb_filter, nb_row, nb_col, **kwargs):
    def f(input):
        layers = [Conv2D(nb_filter // n_group, (nb_row, nb_col))
            (splittensor(axis=1,
                ratio_split=n_group,
                id_split=i)(input))
                    for i in range(n_group)]
        return concatenate(layers, axis=1)

    return f


class Softmax4D(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape


def AlexNet(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             classes=1000, trainable=True):
    """
    Instantiates the AlexNet architecture.
    Optionally loads weights pre-trained
    on ImageNet.
    # Arguments
            include_top: whether to include the fully-connected
                    layer at the top of the network.
            weights: one of `None` (random initialization),
                    'imagenet' (pre-training on ImageNet),
                    or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                    to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                    if `include_top` is False (otherwise the input shape
                    has to be `(224, 224, 3)` (with `channels_last` data format)
                    or `(3, 224, 224)` (with `channels_first` data format).
                    It should have exactly 3 inputs channels,
                    and width and height should be no smaller than 197.
                    E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
    # Returns
            A Keras model instance.
    # Raises
            ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """
    # Input checks
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                 '`None` (random initialization), `imagenet` '
                 '(pre-training on ImageNet), '
                 'or the path to the weights file to be loaded.')
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                 ' as true, `classes` should be 1000')
     
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                    default_size=227,
                                    min_size=197,
                                    data_format='channels_first',
                                    require_flatten=include_top,
                                    weights=weights)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor           
                
    # Define the AlexNet architecture
    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu',
                           name='conv_1')(img_input)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = concatenate([
                       Conv2D(128, (5, 5), activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], axis=1, name='conv_2')

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = concatenate([
                       Conv2D(192, (3, 3), activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], axis=1, name='conv_4')

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = concatenate([
                       Conv2D(128, (3, 3), activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], axis=1, name='conv_5')

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    prediction = Dense(4096, activation='relu', name='dense_2')(dense_2)
    
    # Decide whether output a classification or the last logits
    if include_top:
        prediction = Dropout(0.5)(prediction)
        prediction = Dense(classes, activation='softmax', name='dense_3')(prediction)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model
    model = Model(inputs=inputs, outputs=prediction, name='alexnet')

    # Load weights
    if weights == 'imagenet':
        weights_url = 'http://files.heuritech.com/weights/alexnet_weights.h5'
        weights_path = get_file('alexnet_weights.h5',
                                weights_url,
                                md5_hash='bd0dd1af3674d5b0b39fb627054789b8')
        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights)
        
    # Eventually set the model to untrainable
    if not trainable:
        for layer in model.layers:
            layer.trainable = False

    return model

if __name__ == '__main__':
    target_size = (227,227)
    K.set_image_data_format('channels_first')
    model = AlexNet(include_top=False,
                weights='imagenet',
                input_shape=(3,target_size[0],target_size[1]),
                trainable=False)
    model = Model(inputs=model.inputs[0], outputs=Dense(10, activation='sigmoid', name='dense_last')(model.outputs[0]), name='alexnet')
    model.summary()
    try:
        from keras.utils import plot_model
        plot_model(model, to_file='/Users/MacD/Downloads/model.png')
    except:
        print('Model not visualized')