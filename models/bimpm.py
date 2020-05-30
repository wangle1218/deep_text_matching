#encoding=utf8
import keras
from keras.models import Model
import tensorflow as tf 
import numpy as np 
import sys
sys.path.append( '../')
from engine.base_model import BaseModel
from engine.layers import MultiPerspective

class BiMPM(BaseModel):
    """docstring for BiMPM"""

    def build(self):
        input_left, input_right = self._make_inputs()

        # ----- Embedding layer ----- 
        embedding = self.make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        # ----- Context Representation Layer ----- 
        # rep_left = keras.layers.Bidirectional(keras.layers.LSTM(
        #     self._params['lstm_units'],
        #     return_sequences=True,
        #     dropout=self._params['dropout_rate']
        # ))(embed_left)
        # rep_right = keras.layers.Bidirectional(keras.layers.LSTM(
        #     self._params['lstm_units'],
        #     return_sequences=True,
        #     dropout=self._params['dropout_rate']
        # ))(embed_right)

        bilstm = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))
        rep_left = bilstm(embed_left)
        rep_right = bilstm(embed_right)

        # ----- Matching Layer -----
        matching_layer = MultiPerspective(self._params['mp_dim'])
        matching_left = matching_layer([rep_left, rep_right])
        matching_right = matching_layer([rep_right, rep_left])

        # ----- Aggregation Layer -----
        agg_left = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=False,
            dropout=self._params['dropout_rate']
        ))(matching_left)
        agg_right = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=False,
            dropout=self._params['dropout_rate']
        ))(matching_right)

        aggregation = keras.layers.concatenate([agg_left, agg_right])
        aggregation = keras.layers.Dropout(rate=self._params['dropout_rate'])(aggregation)

        # ----- Prediction Layer -----
        inputs = [input_left, input_right]
        x_out = self._make_output_layer()(aggregation)

        model = keras.Model(inputs=inputs, outputs=x_out)

        return model