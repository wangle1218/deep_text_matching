import pandas as pd
import os
from utils.data_utils import shuffle, pad_sequences
import jieba
import re
from gensim.models import Word2Vec
import numpy as np


# 加载字典
def load_char_vocab():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './input/word2vec/char_vocab.txt')
    vocab = [line.strip() for line in open('./input/word2vec/char_vocab.txt', encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab,start=1)}
    idx2word = {index: word for index, word in enumerate(vocab,start=1)}
    return word2idx, idx2word
# 加载字典
# def load_char_vocab():
#     path = os.path.join(os.path.dirname(__file__), '../input/vocab.txt')
#     vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
#     word2idx = {word: index for index, word in enumerate(vocab)}
#     idx2word = {index: word for index, word in enumerate(vocab)}
#     return word2idx, idx2word

# 加载词典
def load_word_vocab():
    path = os.path.join(os.path.dirname(__file__), '../input/word2vec/word_vocab.txt')
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab,start=1)}
    idx2word = {index: word for index, word in enumerate(vocab,start=1)}
    return word2idx, idx2word

def load_word_embed(nb_words, embed_size):
    EMBEDDING_FILE = './input/word2vec/word2vec.bin'
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = np.mean(all_embs) ,np.std(all_embs)
    print(emb_mean,emb_std)

    vocab = [line.strip() for line in open('./input/word2vec/word_vocab.txt', encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab,start=1)}
    word2idx['PAD'] = 0
    word2idx['UNK'] = len(word2idx)

    embedding_matrix = np.random.normal(emb_mean,emb_std, (nb_words, embed_size))
    
    for word, i in word2idx.items():
        if i >= nb_words: continue
        try:
            embedding_vector = embeddings_index.get(word)
            if not np.mean(embedding_vector): print(embedding_vector)
            embedding_matrix[i] = embedding_vector
        except:
            pass
    embedding_matrix[0] = np.zeros(embed_size)
    print(np.mean(embedding_matrix),np.std(embedding_matrix))

    return embedding_matrix

def load_char_embed(nb_words, embed_size):
    EMBEDDING_FILE = './input/word2vec/char2vec.bin'
    def get_coefs(word,*arr):
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = np.mean(all_embs) ,np.std(all_embs)

    vocab = [line.strip() for line in open('./input/word2vec/char_vocab.txt', encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab,start=1)}
    word2idx['PAD'] = 0
    word2idx['UNK'] = len(word2idx)

    embedding_matrix = np.random.normal(emb_mean,emb_std, (nb_words, embed_size))
    print(embedding_matrix.shape)
    for word, i in word2idx.items():
        if i >= nb_words: continue
        try:
            embedding_vector = embeddings_index.get(word)
            if not np.mean(embedding_vector): print(embedding_vector)
            embedding_matrix[i] = embedding_vector
        except:
            pass
    embedding_matrix[0] = np.zeros(embed_size)
    print(np.mean(embedding_matrix),np.std(embedding_matrix))

    return embedding_matrix


# 字->index
def char_index(p_sentences, h_sentences, maxlen=35):
    word2idx, idx2word = load_char_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=maxlen)
    h_list = pad_sequences(h_list, maxlen=maxlen)

    return p_list, h_list


# 词->index
def word_index(p_sentences, h_sentences,maxlen=15):
    word2idx, idx2word = load_word_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=maxlen)
    h_list = pad_sequences(h_list, maxlen=maxlen)

    return p_list, h_list

# 加载char_index训练数据
def load_char_data(file, data_size=None,maxlen=35):
    path = os.path.join(os.path.dirname(__file__), '../' + file)
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    # [1,2,3,4,5] [4,1,5,2,0]
    p_c_index, h_c_index = char_index(p, h,maxlen=maxlen)

    return p_c_index, h_c_index, label


# 加载char_index、静态词向量、动态词向量的训练数据
def load_all_data(path, data_size=None, maxlen=35):
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_c_index, h_c_index = char_index(p, h, maxlen=maxlen)

    p_seg = list(map(lambda x: list(jieba.cut(x)), p))
    h_seg = list(map(lambda x: list(jieba.cut(x)), h))

    p_w_index, h_w_index = word_index(p_seg, h_seg, maxlen=maxlen)

    # 判断是否有相同的词
    same_word = []
    for p_i, h_i in zip(p_w_index, h_w_index):
        dic = {}
        for i in p_i:
            if i == 0:
                break
            dic[i] = dic.get(i, 0) + 1
        for index, i in enumerate(h_i):
            if i == 0:
                same_word.append(0)
                break
            dic[i] = dic.get(i, 0) - 1
            if dic[i] == 0:
                same_word.append(1)
                break
            if index == len(h_i) - 1:
                same_word.append(0)

    return p_c_index, h_c_index, p_w_index, h_w_index, same_word, label


if __name__ == '__main__':
    load_all_data('./input/train.csv', data_size=100)