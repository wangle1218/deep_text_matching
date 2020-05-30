from gensim.models import Word2Vec
import pandas as pd
import jieba

df = pd.read_csv('input/train.csv')
p = df['sentence1'].values
h = df['sentence2'].values
p_seg = list(map(lambda x: list(jieba.cut(x.replace(" ",""))), p))
h_seg = list(map(lambda x: list(jieba.cut(x.replace(" ",""))), h))
common_texts = []
common_texts.extend(p_seg)
common_texts.extend(h_seg)

df = pd.read_csv('input/dev.csv')
p = df['sentence1'].values
h = df['sentence2'].values
p_seg = list(map(lambda x: list(jieba.cut(x.replace(" ",""))), p))
h_seg = list(map(lambda x: list(jieba.cut(x.replace(" ",""))), h))
common_texts.extend(p_seg)
common_texts.extend(h_seg)

df = pd.read_csv('input/test.csv')
p = df['sentence1'].values
h = df['sentence2'].values
p_seg = list(map(lambda x: list(jieba.cut(x.replace(" ",""))), p))
h_seg = list(map(lambda x: list(jieba.cut(x.replace(" ",""))), h))
common_texts.extend(p_seg)
common_texts.extend(h_seg)
model = Word2Vec(common_texts, size=100, window=5, min_count=3, workers=12)

model.save("input/word2vec/word2vec.model")
model.wv.save_word2vec_format('input/word2vec/word2vec.bin',binary=False) 
word_set = set()
for sample in common_texts:
    for word in sample:
        word_set.add(word)
with open('input/word2vec/word_vocab.txt','w',encoding='utf8') as f:
    f.write("\n".join(sorted(list(word_set),reverse=True)))

p_seg = list(map(lambda x: list(x.replace(" ","")), p))
h_seg = list(map(lambda x: list(x.replace(" ","")), h))
common_texts = []
common_texts.extend(p_seg)
common_texts.extend(h_seg)

df = pd.read_csv('input/dev.csv')
p = df['sentence1'].values
h = df['sentence2'].values
p_seg = list(map(lambda x: list(x.replace(" ","")), p))
h_seg = list(map(lambda x: list(x.replace(" ","")), h))
common_texts.extend(p_seg)
common_texts.extend(h_seg)

df = pd.read_csv('input/test.csv')
p = df['sentence1'].values
h = df['sentence2'].values
p_seg = list(map(lambda x: list(x.replace(" ","")), p))
h_seg = list(map(lambda x: list(x.replace(" ","")), h))
common_texts.extend(p_seg)
common_texts.extend(h_seg)
model = Word2Vec(common_texts, size=100, window=5, min_count=3, workers=12)

model.save("input/word2vec/char2vec.model")
model.wv.save_word2vec_format('input/word2vec/char2vec.bin',binary=False) 
char_set = set()
for sample in common_texts:
    for char in sample:
        char_set.add(char)
with open('input/word2vec/char_vocab.txt','w',encoding='utf8') as f:
    f.write("\n".join(sorted(list(char_set),reverse=True)))