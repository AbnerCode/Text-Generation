# -*- coding:utf-8 -*- 
import pandas as pd
import numpy as np
import csv
import spacy
from gensim.models import word2vec
import tqdm
import tensorflow as tf

#obj = pd.read_csv('data.csv')
vocab = [] #抽取动词和名词
nlp = spacy.load('en_core_web_sm')
with open('data.csv','r') as csvfile:
    reader = csv.reader(csvfile)  
    for i,rows in enumerate(reader):
        if i == 0:
            text = rows
            str_convert = ''.join(text)

#print(type(str_convert))
# Process whole documents
doc = nlp(str_convert)
# Find named entities, phrases and concepts

for token in doc:
	if token.pos_ == 'VERB' or token.pos_ == 'NOUN':
		vocab.append(token.text)
'''
with open('train-data.txt','r',encoding="utf-8") as f:
   with open('train-data-seg.txt','w',encoding="utf-8") as f1: 
     for line in f:
         #print(line)
         line = jieba.cut(line)
         line1 = []
         for i in line :
             line1.append(i)
         line1 = sentence_cut (line1) + ' '
#         print(line1)
         sentences.append(line1)
         f1.write(line1)
f.close()
f1.close()

with open('train-data-seg.txt','w',encoding = "utf-8") as f:
	data = []
	for token in doc:
		if token.pos_ == 'VERB' or token.pos_ == 'NOUN':
			data.append(token.text)
			data_str = ' '.join(data)
	f.write(data_str)			
f.close()

with open('train-data.txt','w',encoding = "utf-8") as f:
	data = []
	for token in doc:
		if token.pos_ == 'VERB' or token.pos_ == 'NOUN':
			data.append(token.text)
			data_str = '\n'.join(data)
	f.write(data_str)			
f.close()
'''
#加载glove
# 加载预训练好的glove词向量

# 单词到编码的映射，例如machine -> 10283
word_to_token = {word: token for token, word in enumerate(vocab)}
# 编码到单词的映射，例如10283 -> machine
token_to_word = {token: word for word, token in word_to_token.items()}

with open("glove.6B.50d.txt", 'r') as f:
    words = set()
    word_to_vec = {}
    for line in f:
        line = line.strip().split()
        # 当前单词
        curr_word = line[0]
        words.add(curr_word)
        # 当前词向量
        word_to_vec[curr_word] = np.array(line[1:], dtype=np.float32)

VOCAB_SIZE = len(vocab)  # 10384
EMBEDDING_SIZE = 50

# 初始化词向量矩阵（这里命名为static是因为这个词向量矩阵用预训练好的填充，无需重新训练）
static_embeddings = np.zeros([VOCAB_SIZE, EMBEDDING_SIZE])

for word, token in tqdm.tqdm(word_to_token.items()):
    # 用glove词向量填充，如果没有对应的词向量，则用随机数填充
    word_vector = word_to_vec.get(word, 0.2 * np.random.random(EMBEDDING_SIZE) - 0.1)
    static_embeddings[token, :] = word_vector

static_embeddings = static_embeddings.astype(np.float32)
#print (static_embeddings.shape)

all_static_embeddings = tf.reshape(static_embeddings,[700,1])
#def buid_lstm():


