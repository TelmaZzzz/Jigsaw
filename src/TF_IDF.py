import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import re 
import scipy
from scipy import sparse

from pprint import pprint
from matplotlib import pyplot as plt 

import time
import scipy.optimize as optimize
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_colwidth=300
pd.options.display.max_columns = 100

from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.linear_model import Ridge

import queue
import copy
import logging
logging.getLogger().setLevel(logging.DEBUG)


def text_cleaning(text):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis
    
    text - Text piece to be cleaned.
    '''
    template = re.compile(r'https?://\S+|www\.\S+') #Removes website links
    text = template.sub(r'', text)
    
    soup = BeautifulSoup(text, 'lxml') #Removes HTML tags
    only_text = soup.get_text()
    text = only_text
    
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    text = re.sub(r"[^a-zA-Z\d]", " ", text) #Remove special Charecters
    text = re.sub(' +', ' ', text) #Remove Extra Spaces
    text = text.strip() # remove spaces at the beginning and at the end of string

    return text


def hash(mp):
    cat_list = ['obscene', 'toxic', 'threat', 'insult', 'severe_toxic', 'identity_hate']
    res = 0
    for item in cat_list:
        res*=400
        res+=(int(mp[item]*100))
    return res


def rehash(h):
    cat_list = ['obscene', 'toxic', 'threat', 'insult', 'severe_toxic', 'identity_hate']
    cat_list.reverse()
    res = h
    mp = dict()
    for item in cat_list:
        mp[item] = h%400 / 100
        h //= 400
    return mp

def cal(df_train, df_val):
    min_len = (df_train['y'] > 0).sum()  # len of toxic comments
    df_y0_undersample = df_train[df_train['y'] == 0].sample(n=min_len, random_state=201)  # take non toxic comments
    df = pd.concat([df_train[df_train['y'] > 0], df_y0_undersample])  # make new df

    vec = TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))
    X = vec.fit_transform(df['text'])
    logging.info(X)
    model = Ridge(alpha=0.2)
    model.fit(X, df['y'])

    X_less_toxic = vec.transform(df_val['less_toxic'])
    X_more_toxic = vec.transform(df_val['more_toxic'])

    p1 = model.predict(X_less_toxic)
    p2 = model.predict(X_more_toxic)
    return (p1 < p2).mean()


def bfs():
    df_train = pd.read_csv("/users10/lyzhang/opt/tiger/jigsaw/data/jigsaw_cls/train.csv")
    df_train = df_train.rename(columns={'comment_text':'text'})
    df_val = pd.read_csv("/users10/lyzhang/opt/tiger/jigsaw/data/validation_data.csv")
    df_train['text'] = df_train['text'].apply(text_cleaning)
    df_val['less_toxic'] = df_val['less_toxic'].apply(text_cleaning)
    df_val['more_toxic'] = df_val['more_toxic'].apply(text_cleaning)
    cat_mtpl = {'obscene': 0.16, 'toxic': 0.32, 'threat': 1.5, 
                'insult': 0.64, 'severe_toxic': 0.64, 'identity_hate': 1.5}
    cat_list = ['obscene', 'toxic', 'threat', 'insult', 'severe_toxic', 'identity_hate']
    q = queue.PriorityQueue()
    vis = set([hash(cat_mtpl)])
    q.put([0, hash(cat_mtpl)])
    res_list = []
    df_train['score'] = df_train.loc[:, 'toxic':'identity_hate'].sum(axis=1)
    min_len = (df_train['score'] > 0).sum()
    df_y0_undersample = df_train[df_train['score'] == 0].sample(n=min_len, random_state=201)
    df_train = pd.concat([df_train[df_train['score'] > 0], df_y0_undersample])
    while q.empty() is False:
        hash_mtpl = q.get()[1]
        mtpl = rehash(hash_mtpl)
        df = copy.deepcopy(df_train)
        for c in mtpl:
            df[c] = df[c] * mtpl[c]
        df['score'] = df.loc[:, 'toxic':'identity_hate'].sum(axis=1)
        df['y'] = df['score']
        acc = cal(df, df_val)
        res_list.append((acc, mtpl))
        logging.info(f"acc: {acc}, mtpl:{mtpl}")
        for c in cat_list:
            mtpl[c] = mtpl[c] + 0.02
            if mtpl[c] <= 2 and mtpl[c] >= 0 and (hash(mtpl) not in vis):
                vis.add(hash(mtpl))
                q.put([-acc, hash(mtpl)])
            mtpl[c] = mtpl[c] - 0.02

            mtpl[c] = mtpl[c] - 0.02
            if mtpl[c] <= 2 and mtpl[c] >= 0 and (hash(mtpl) not in vis):
                vis.add(hash(mtpl))
                q.put([-acc, hash(mtpl)])
            mtpl[c] = mtpl[c] + 0.02
    res_list = sorted(res_list, reverse=True)
    logging.info("END...")
    for acc, mtpl in res_list[:10]:
        logging.info(f"acc: {acc}, mtpl:{mtpl}")


def main():
    logging.info("Start Load")
    df_train = pd.read_csv("/users10/lyzhang/opt/tiger/jigsaw/data/jigsaw_cls/train.csv")
    df_train = df_train.rename(columns={'comment_text':'text'})
    df_val = pd.read_csv("/users10/lyzhang/opt/tiger/jigsaw/data/validation_data.csv")
    # df_train['text'] = df_train['text'].apply(text_cleaning)
    # df_val['less_toxic'] = df_val['less_toxic'].apply(text_cleaning)
    # df_val['more_toxic'] = df_val['more_toxic'].apply(text_cleaning)
    df_train['y'] = df_train.loc[:, 'toxic':'identity_hate'].sum(axis=1)
    min_len = (df_train['y'] > 0).sum()  # len of toxic comments
    df_y0_undersample = df_train[df_train['y'] == 0].sample(n=min_len, random_state=201)  # take non toxic comments
    df = pd.concat([df_train[df_train['y'] > 0], df_y0_undersample])  # make new df
    logging.info("Finish Load")
    # df_train['text'] = df_train['text'].apply(text_cleaning)
    # df_val['less_toxic'] = df_val['less_toxic'].apply(text_cleaning)
    # df_val['more_toxic'] = df_val['more_toxic'].apply(text_cleaning)
    vec = TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5), max_features = 46000)
    X = vec.fit_transform(df['text'])
    logging.info("Finish fit")
    Y = vec.transform(df_val['more_toxic'])
    logging.info(Y.shape)
    vsm = Y.toarray()
    category_keywords_li = []
    logging.info(vsm.shape)
    for i in tqdm(range(vsm.shape[0])):
        sorted_keyword = sorted(zip(vec.get_feature_names(), vsm[i]), key=lambda x:x[1], reverse=True)
        category_keywords = [w[0] for w in sorted_keyword[:10]]
        category_keywords_li.append(category_keywords)
    logging.info(category_keywords_li[:10])


# bfs()
main()