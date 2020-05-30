import keras
from keras.models import Model
import tensorflow as tf 
import numpy as np 
import sys
import os
sys.path.append( '../')
from engine.base_model import BaseModel

np.random.seed(1)
tf.set_random_seed(1)


class CDSSM(BaseModel):

    def _create_base_network(self):

        def _wrapper(x):

            x = self.embedding(x)
            x = keras.layers.Conv1D(
                filters=self._params['filters'],
                kernel_size=self._params['kernel_size'],
                strides=self._params['strides'],
                padding=self._params['padding'],
                activation=self._params['conv_activation_func'],
                kernel_initializer=self._params['w_initializer'],
                bias_initializer=self._params['b_initializer'])(x)
            # Apply max pooling by take max at each dimension across
            # all word_trigram features.
            x = keras.layers.Dropout(self._params['dropout_rate'])(x)
            x = keras.layers.GlobalMaxPool1D()(x)
            # Apply a none-linear transformation use a tanh layer.
            x = self._make_multi_layer_perceptron_layer()(x)
            return x

        return _wrapper

    def build(self):
        """
        Build model structure.
        CDSSM use Siamese architecture.
        """
        self.embedding = self.make_embedding_layer()
        base_network = self._create_base_network()
        # Left input and right input.
        input_left, input_right = self._make_inputs()
        # Process left & right input.
        x = [base_network(input_left),
             base_network(input_right)]
        # Dot product with cosine similarity.
        x = keras.layers.Dot(axes=[1, 1], normalize=True)(x)
        x_out = self._make_output_layer()(x)
        model = Model(inputs=[input_left, input_right],
                              outputs=x_out)
        return model