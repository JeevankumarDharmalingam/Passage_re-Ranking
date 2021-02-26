# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# %% [code]
import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util, InputExample
import time, gzip, os, torch
import pandas as pd
from torch.utils.data import DataLoader
import random, math
import nltk
import re

nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# %% [code]
import sys

df_question = pd.read_csv(r"C:\Users\Jeev\Downloads\Compressed\usecaseassesstementeygds/questions.csv")
df_para = pd.read_csv(r"C:\Users\Jeev\Downloads\Compressed\usecaseassesstementeygds/paragraphs.csv", names=range(6), encoding='unicode_escape', delimiter="\t")
df_train = pd.read_csv(r"C:\Users\Jeev\Downloads\Compressed\usecaseassesstementeygds/train.csv")
headers = df_para.iloc[0]
df_para = pd.DataFrame(df_para.values[1:], columns=headers)
df_para = df_para.fillna("")
df_para['paragraph'] = df_para[['Title', 'SectionTitle', 'SubsectionTitle', 'ParaText']].apply(lambda x: " ".join(x),
                                                                                               axis=1)
df_para.drop(0, axis=0, inplace=True)
df_para = df_para.reset_index(drop=True)

# %% [code]
from tqdm import tqdm

lemma = WordNetLemmatizer()
corpus = []

para_id = []
corpus = []
for i in tqdm(range(1, len(df_para))):
    rev = re.sub('[^a-zA-Z0-9]', ' ', df_para.loc[i, 'paragraph'])
    rev = rev.lower()
    rev = rev.split()

    rev = [lemma.lemmatize(words) for words in rev if not words in stopwords.words('english')]
    rev = ' '.join(rev)
    para_id.append(df_para.loc[i, 'ParaId'])
    corpus.append(rev)

para_df = pd.DataFrame(list(zip(para_id, corpus)), columns=["Para_id", "Corpus"])


# %% [code]
def qid_to_question(qid):
    return df_question[df_question["qid"] == qid]["qtext"].values[0]


def pid_to_para(pid):
    return para_df[para_df["Para_id"] == str(pid)]["Corpus"].values[0]


# %% [code]
df_train["question"] = df_train["qid"].map(qid_to_question)
df_train["para"] = df_train["ParaId"].map(pid_to_para)

# %% [code]
model_name = 'msmarco-roberta-base-v2'
bi_encoder = SentenceTransformer(model_name)
corpus_embeddings = bi_encoder.encode(corpus, convert_to_tensor=True, show_progress_bar=True)


# %% [code]
def negative_sentence(query):
    question_embedding = bi_encoder.encode(query.lower(), convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=1000)
    hits = hits[0]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    hit = hits[len(hits) - 1]
    neg_passage = corpus[hit['corpus_id']].replace("\n", " ")
    return neg_passage


# %% [code]
df_train["Negative_para"] = df_train['question'].apply(lambda x: negative_sentence(x))

# %% [code]
train_samples = []
for query, para, neg_para in zip(df_train['question'], df_train['para'], df_train['Negative_para']):
    train_samples.append(InputExample(texts=[query, para], label=(random.randint(90, 98)) / 100))
    train_samples.append(InputExample(texts=[query, neg_para], label=(random.randint(50, 220)) / 1000))

# %% [code]
num_epochs = 50
model_save_path = 'output/training_bench.h5'
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=1)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

# %% [code]
model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', num_labels=1)

# %% [code]
model.fit(train_dataloader=train_dataloader,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path="mdel.h5")


# %% [code]
def top_5_sentence(query):
    question_embedding = bi_encoder.encode(query.lower(), convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=100)
    hits = hits[0]
    cross_inp = [[query, corpus[hit['corpus_id']]] for hit in hits]
    cross_scores = model.predict(cross_inp)
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]
    print("Top-5 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[0:10]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], corpus[hit['corpus_id']].replace("\n", " ")))


# %% [code]
q = "How is an endowment contract defined?"
print(top_5_sentence(q))


