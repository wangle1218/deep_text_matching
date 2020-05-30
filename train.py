#encoding=utf8

import tensorflow as tf 
import keras 
import numpy as np 
import os
import sys
sys.path.append('../')
from models.cdssm import CDSSM
from models.mvlstm import MVLSTM
from models.arcii import ArcII
from models.bimpm import BiMPM
from models.esim import ESIM
from models.match_pyramid import MatchPyramid
from models.drcn import DRCN
from engine.layers import KMaxPooling,MatchingLayer,MultiPerspective
from utils.load_data import load_char_data,load_word_embed,load_char_embed,load_all_data


np.random.seed(1)
tf.set_random_seed(1)

base_params = {
    'num_classes':2,
    'max_features':1700,
    'embed_size':200,
    'filters':300,
    'kernel_size':3,
    'strides':1,
    'padding':'same',
    'conv_activation_func':'relu',
    'embedding_matrix':[],
    'w_initializer':'random_uniform',
    'b_initializer':'zeros',
    'dropout_rate':0.2,
    'mlp_activation_func':'relu',
    'mlp_num_layers':1,
    'mlp_num_units':128,
    'mlp_num_fan_out':128,
    'input_shapes':[(64,),(64,)],
    'task':'Classification',
}

if __name__ == '__main__':
    
    model_name = sys.argv[1]

    if model_name == "cdssm":
        params = base_params
        backend = CDSSM(params)
    elif model_name == "mvlstm":
        mvlstm_params = base_params
        mvlstm_params['lstm_units'] = 64
        mvlstm_params['top_k'] = 50
        mvlstm_params['mlp_num_units'] = 128
        mvlstm_params['mlp_num_fan_out'] = 128
        mvlstm_params['dropout_rate'] = 0.3
        mvlstm_params['embed_size'] = 100
        char_embedding_matrix = load_char_embed(mvlstm_params['max_features'],mvlstm_params['embed_size'])
        mvlstm_params['embedding_matrix'] = char_embedding_matrix
        params = mvlstm_params
        backend = MVLSTM(params)
    elif model_name == "arcii":
        arcii_params = base_params
        arcii_params['matching_type'] = 'dot'
        arcii_params['num_blocks'] = 3
        arcii_params['kernel_1d_count'] = 32
        arcii_params['kernel_1d_size'] = 3
        arcii_params['kernel_2d_count'] = [16, 32, 32]
        arcii_params['kernel_2d_size'] = [[3, 3], [3, 3], [3, 3]]
        arcii_params['pool_2d_size'] = [[2, 2], [2, 2], [2, 2]]
        arcii_params['dropout_rate'] = 0.5
        params = arcii_params
        backend = ArcII(params)
    elif model_name == "bimpm":
        bimpm_params = base_params
        bimpm_params['mp_dim'] = 12
        bimpm_params['lstm_units'] = 64
        bimpm_params['dropout_rate'] = 0.2
        bimpm_params['embed_size'] = 100
        char_embedding_matrix = load_char_embed(bimpm_params['max_features'],bimpm_params['embed_size'])
        bimpm_params['embedding_matrix'] = char_embedding_matrix
        params = bimpm_params
        backend = BiMPM(params)
    elif model_name == "esim":
        esim_params = base_params
        esim_params['mlp_num_layers'] = 1
        esim_params['mlp_num_units'] = 256
        esim_params['mlp_num_fan_out'] = 128
        esim_params['lstm_units'] = 64
        esim_params['dropout_rate'] = 0.3
        esim_params['embed_size'] = 100
        char_embedding_matrix = load_char_embed(esim_params['max_features'],esim_params['embed_size'])
        esim_params['embedding_matrix'] = char_embedding_matrix
        params = esim_params
        backend = ESIM(params)
    elif model_name == "match_pyramid":
        mp_params = base_params
        mp_params['matching_type'] = 'dot'
        mp_params['num_blocks'] = 2
        mp_params['kernel_count'] = [16, 32]
        mp_params['kernel_size'] = [[3, 3], [3, 3]]
        mp_params['pool_size'] = [3, 3]
        mp_params['mlp_num_layers'] = 1
        mp_params['mlp_num_units'] = 128
        mp_params['mlp_num_fan_out'] = 128
        mp_params['embed_size'] = 100
        char_embedding_matrix = load_char_embed(mp_params['max_features'],mp_params['embed_size'])
        mp_params['embedding_matrix'] = char_embedding_matrix
        params = mp_params
        backend = MatchPyramid(params)
    elif model_name == "drcn":
        drcn_params = base_params
        drcn_params['input_shapes'] = [(48,),(48,),(48,),(48,)]
        drcn_params['lstm_units'] = 64
        drcn_params['num_blocks'] = 1
        drcn_params['mlp_num_layers'] = 1
        drcn_params['mlp_num_units'] = 256
        drcn_params['mlp_num_fan_out'] = 128
        drcn_params['max_features'] = 1700
        drcn_params['word_max_features'] = 7300
        drcn_params['word_embed_size'] = 100
        drcn_params['embed_size'] = 100

        word_embedding_matrix = load_word_embed(drcn_params['word_max_features'],drcn_params['word_embed_size'])
        char_embedding_matrix = load_char_embed(drcn_params['max_features'],drcn_params['embed_size'])

        drcn_params['embedding_matrix'] = char_embedding_matrix
        drcn_params['word_embedding_matrix'] = word_embedding_matrix

        params = drcn_params
        backend = DRCN(params)

    if model_name == "drcn":
        p_c_index, h_c_index, p_w_index, h_w_index, same_word, y = load_all_data('./input/train.csv',maxlen=params['input_shapes'][0][0])
        x = [p_c_index, h_c_index, p_w_index, h_w_index]
        y = keras.utils.to_categorical(y,num_classes=params['num_classes'])
        p_c_index_evl, h_c_index_evl, p_w_index_evl, h_w_index_evl, same_word_evl, y_eval = load_all_data('./input/dev.csv',maxlen=params['input_shapes'][0][0])
        x_eval = [p_c_index_evl, h_c_index_evl, p_w_index_evl, h_w_index_evl]
        y_eval = keras.utils.to_categorical(y_eval,num_classes=params['num_classes'])
        p_c_index_test, h_c_index_test, p_w_index_test, h_w_index_test, same_word_test, y_test = load_all_data('./input/test.csv',maxlen=params['input_shapes'][0][0])
        x_test = [p_c_index_test, h_c_index_test, p_w_index_test, h_w_index_test]
        y_test = keras.utils.to_categorical(y_test,num_classes=params['num_classes'])
    else:
        p, h, y = load_char_data('input/train.csv', data_size=None,maxlen=params['input_shapes'][0][0])
        x = [p,h]
        y = keras.utils.to_categorical(y,num_classes=params['num_classes'])
        p_eval, h_eval, y_eval = load_char_data('input/dev.csv', data_size=None,maxlen=params['input_shapes'][0][0])
        x_eval = [p_eval,h_eval]
        y_eval = keras.utils.to_categorical(y_eval,num_classes=params['num_classes'])
        p_test, h_test, y_test = load_char_data('input/test.csv', data_size=None,maxlen=params['input_shapes'][0][0])
        x_test = [p_test,h_test]
        y_test = keras.utils.to_categorical(y_test,num_classes=params['num_classes'])


    model = backend.build()
    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
        )
    print(model.summary())

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=4, 
        verbose=2, 
        mode='max'
        )
    bast_model_filepath = './output/best_%s_model.h5' % model_name
    checkpoint = keras.callbacks.ModelCheckpoint(
        bast_model_filepath, 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True,
        mode='max'
        )
    model.fit(
        x=x, 
        y=y, 
        batch_size=64, 
        epochs=15, 
        validation_data=(x_eval, y_eval), 
        shuffle=True, 
        callbacks=[earlystop,checkpoint]
        )

    model_frame_path = "./output/%s_model.json" % model_name
    model_json = model.to_json()
    with open(model_frame_path, "w") as json_file:
        json_file.write(model_json)

    custom_objects = None
    if model_name == "mvlstm":
        custom_objects = {'KMaxPooling':KMaxPooling}
    elif model_name == "arcii":
        custom_objects = {'MatchingLayer':MatchingLayer}
    elif model_name == "bimpm":
        custom_objects = {'MultiPerspective':MultiPerspective}
    elif model_name == "match_pyramid":
        custom_objects = {'MatchingLayer':MatchingLayer}

    # model = keras.models.model_from_json(
    #     open(model_frame_path,"r").read(),
    #     custom_objects=custom_objects
    #     )
    model.load_weights(bast_model_filepath)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
        )

    loss, acc = model.evaluate(
        x=x_test, 
        y=y_test, 
        batch_size=128, 
        verbose=1
        )
    print("Test loss:",loss, "Test accuracy:",acc)