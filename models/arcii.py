#encoding=utf8
import keras
from keras.models import Model
import tensorflow as tf 
import numpy as np 
import sys
sys.path.append( '../')
from engine.base_model import BaseModel
from engine.layers import KMaxPooling,MatchingLayer


class ArcII(BaseModel):
    """docstring for ArcII"""
    def build(self):
        """
        Build model structure.
        ArcII has the desirable property of letting two sentences meet before
        their own high-level representations mature.
        """
        input_left, input_right = self._make_inputs()

        embedding = self.make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        # Phrase level representations
        # conv_1d_left = keras.layers.Conv1D(
        #     self._params['kernel_1d_count'],
        #     self._params['kernel_1d_size'],
        #     padding=self._params['padding']
        # )(embed_left)
        # conv_1d_right = keras.layers.Conv1D(
        #     self._params['kernel_1d_count'],
        #     self._params['kernel_1d_size'],
        #     padding=self._params['padding']
        # )(embed_right)
        conv_1d = keras.layers.Conv1D(
            self._params['kernel_1d_count'],
            self._params['kernel_1d_size'],
            padding=self._params['padding']
        )
        conv_1d_left = conv_1d(embed_left)
        conv_1d_right = conv_1d(embed_right)

        # Interaction
        embed_cross = MatchingLayer(
            normalize=True,
            matching_type=self._params['matching_type']
            )([conv_1d_left, conv_1d_right])

        for i in range(self._params['num_blocks']):
            embed_cross = self._conv_pool_block(
                embed_cross,
                self._params['kernel_2d_count'][i],
                self._params['kernel_2d_size'][i],
                self._params['padding'],
                self._params['conv_activation_func'],
                self._params['pool_2d_size'][i]
            )

        embed_flat = keras.layers.Flatten()(embed_cross)
        x = keras.layers.Dropout(rate=self._params['dropout_rate'])(embed_flat)

        inputs = [input_left, input_right]
        x_out = self._make_output_layer()(x)
        model = keras.Model(inputs=inputs, outputs=x_out)

        return model

    @classmethod
    def _conv_pool_block(cls, x,kernel_count, kernel_size,padding,activation,pool_size):
        output = keras.layers.Conv2D(kernel_count,
                                     kernel_size,
                                     padding=padding,
                                     activation=activation)(x)
        output = keras.layers.MaxPooling2D(pool_size=pool_size)(output)
        # output = keras.layers.normalization.BatchNormalization()(output)
        return output