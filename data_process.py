# -*- coding:utf-8 -*- 
#抽取名次动词的update
import pandas as pd
import numpy as np
import csv
import spacy
import tqdm
import tensorflow as tf


vocab = [] #抽取动词和名词
nlp = spacy.load('en_core_web_sm')
with open('data.csv','r') as csvfile:
    reader = csv.reader(csvfile)  
    for i,rows in enumerate(reader):
        if i < 45490:
            text = rows
            str_convert = ''.join(text)
            doc = nlp(str_convert)
            data = []
            for token in doc:
            	if token.pos_ == 'VERB' or token.pos_ == 'NOUN':
            		data.append(token.text)
            		data_str = ' '.join(data)
            with open('train-data-seg.txt','a',encoding = "utf-8") as f:
            	f.write(data_str+'\n')
            		




