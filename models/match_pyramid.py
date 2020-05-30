#encoding=utf8
import keras
from keras.models import Model
import tensorflow as tf 
import numpy as np 
import sys
sys.path.append( '../')
from engine.base_model import BaseModel
from engine.layers import MatchingLayer


class MatchPyramid(BaseModel):
    """
    MatchPyramid Model.
    """

    def build(self):
        """
        Build model structure.
        MatchPyramid text matching as image recognition.
        """
        input_left, input_right = self._make_inputs()

        embedding = self.make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        # Interaction
        matching_layer = MatchingLayer(matching_type=self._params['matching_type'])
        embed_cross = matching_layer([embed_left, embed_right])

        for i in range(self._params['num_blocks']):
            embed_cross = self._conv_block(
                embed_cross,
                self._params['kernel_count'][i],
                self._params['kernel_size'][i],
                self._params['padding'],
                self._params['conv_activation_func']
            )

        embed_pool = keras.layers.MaxPooling2D(pool_size=self._params['pool_size'])(embed_cross)

        embed_flat = keras.layers.Flatten()(embed_pool)
        mlp = self._make_multi_layer_perceptron_layer()(embed_flat)
        mlp = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(mlp)

        inputs = [input_left, input_right]
        prediction = self._make_output_layer()(mlp)
        model = keras.Model(inputs=inputs, outputs=prediction)

        return model

    @classmethod
    def _conv_block(cls, x,kernel_count,kernel_size, padding,activation ):
        output = keras.layers.Conv2D(kernel_count,
                                     kernel_size,
                                     padding=padding,
                                     activation=activation)(x)
        return output