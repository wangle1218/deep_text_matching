#encoding=utf8
import abc
import keras
from keras.models import Model
import tensorflow as tf 
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)


class BaseModel(object):

    def __init__( self, params):
        """Init."""
        self._params = params

    def _make_embedding_layer( self,name,**kwargs) -> keras.layers.Layer:
        return keras.layers.Embedding(
            self._params['max_features'],
            self._params['embed_size'],
            trainable=False,
            name=name,
            **kwargs
        )

    def make_embedding_layer(self,name='embedding',**kwargs):
        if self._params['embedding_matrix']:
            embedding = self._make_embedding_layer(name=name,**kwargs)
            embedding.set_weights([self._params['embedding_matrix']])
        else:
            embedding = self._make_embedding_layer(name=name,**kwargs)

        return embedding

    def _make_multi_layer_perceptron_layer(self) -> keras.layers.Layer:
        # TODO: do not create new layers for a second call
        def _wrapper(x):
            activation = self._params['mlp_activation_func']
            for _ in range(self._params['mlp_num_layers']):
                x = keras.layers.Dense(self._params['mlp_num_units'],
                                       activation=activation)(x)
            return keras.layers.Dense(self._params['mlp_num_fan_out'],
                                      activation=activation)(x)

        return _wrapper

    def _make_inputs(self) -> list:
        input_left = keras.layers.Input(
            name='text_left',
            shape=self._params['input_shapes'][0]
        )
        input_right = keras.layers.Input(
            name='text_right',
            shape=self._params['input_shapes'][1]
        )
        return [input_left, input_right]

    def _make_output_layer(self) -> keras.layers.Layer:
        """:return: a correctly shaped keras dense layer for model output."""
        task = self._params['task']
        if task == "Classification":
            return keras.layers.Dense(self._params['num_classes'], activation='softmax')
        elif task == "Ranking":
            return keras.layers.Dense(1, activation='linear')
        else:
            raise ValueError(f"{task} is not a valid task type."
                             f"Must be in `Ranking` and `Classification`.")

    def _create_base_network(self):

        def _wrapper(x):

            pass

        return _wrapper

    def build(self):
        """
        Build model structure.
        """
        pass
        return model



