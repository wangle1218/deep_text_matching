#encoding=utf8
import keras
from keras.models import Model
import tensorflow as tf 
import numpy as np 
import sys
sys.path.append( '../')
from engine.base_model import BaseModel
from engine.layers import SoftAttention


class ESIM(BaseModel):
    """docstring for ESIM"""
    def build(self):
        """
        Build the model.
        """
        a, b = self._make_inputs()

        # ---------- Embedding layer ---------- #
        embedding = self.make_embedding_layer()
        embedded_a = embedding(a)
        embedded_b = embedding(b)

        # ---------- Encoding layer ---------- #
        # encoded_a = keras.layers.Bidirectional(keras.layers.LSTM(
        #     self._params['lstm_units'],
        #     return_sequences=True,
        #     dropout=self._params['dropout_rate']
        # ))(embedded_a)
        # encoded_b = keras.layers.Bidirectional(keras.layers.LSTM(
        #     self._params['lstm_units'],
        #     return_sequences=True,
        #     dropout=self._params['dropout_rate']
        # ))(embedded_b)

        bilstm = keras.layers.Bidirectional(keras.layers.LSTM(
                    self._params['lstm_units'],
                    return_sequences=True,
                    dropout=self._params['dropout_rate']
                ))

        encoded_a = bilstm(embedded_a)
        encoded_b = bilstm(embedded_b)

        # ---------- Local inference layer ---------- #
        atten_a, atten_b = SoftAttention()([encoded_a, encoded_b])

        sub_a_atten = keras.layers.Lambda(lambda x: x[0]-x[1])([encoded_a, atten_a])
        sub_b_atten = keras.layers.Lambda(lambda x: x[0]-x[1])([encoded_b, atten_b])

        mul_a_atten = keras.layers.Lambda(lambda x: x[0]*x[1])([encoded_a, atten_a])
        mul_b_atten = keras.layers.Lambda(lambda x: x[0]*x[1])([encoded_b, atten_b])

        m_a = keras.layers.concatenate([encoded_a, atten_a, sub_a_atten, mul_a_atten])
        m_b = keras.layers.concatenate([encoded_b, atten_b, sub_b_atten, mul_b_atten])

        # ---------- Inference composition layer ---------- #
        composition_a = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))(m_a)

        avg_pool_a = keras.layers.GlobalAveragePooling1D()(composition_a)
        max_pool_a = keras.layers.GlobalMaxPooling1D()(composition_a)

        composition_b = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))(m_b)

        avg_pool_b = keras.layers.GlobalAveragePooling1D()(composition_b)
        max_pool_b = keras.layers.GlobalMaxPooling1D()(composition_b)

        pooled = keras.layers.concatenate([avg_pool_a, max_pool_a, avg_pool_b, max_pool_b])
        pooled = keras.layers.Dropout(rate=self._params['dropout_rate'])(pooled)

        # ---------- Classification layer ---------- #
        mlp = self._make_multi_layer_perceptron_layer()(pooled)
        mlp = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(mlp)

        prediction = self._make_output_layer()(mlp)

        model = Model(inputs=[a, b], outputs=prediction)

        return model
        