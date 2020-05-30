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

    
    def make_embedding_layer(self,name='embedding',embed_type='char',**kwargs):

        def init_embedding(weights=None):
            if embed_type == "char":
                input_dim = self._params['max_features']
                output_dim = self._params['embed_size']
            else:
                input_dim = self._params['word_max_features']
                output_dim = self._params['word_embed_size']

            return keras.layers.Embedding(
                input_dim = input_dim,
                output_dim = output_dim,
                trainable = False,
                name = name,
                weights = weights,
                **kwargs)

        if embed_type == "char":
            embed_weights = self._params['embedding_matrix']
        else:
            embed_weights = self._params['word_embedding_matrix']

        if embed_weights == []:
            embedding = init_embedding()
        else:
            embedding = init_embedding(weights = [embed_weights])

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



