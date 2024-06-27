import re
import numpy as np
from tqdm import tqdm
import math
import MeCab
import json
import gensim
import pyLDAvis
import pyLDAvis.gensim
#from wordcloud import WordCloud
import matplotlib.pylab as plt
import os, sys
from sudachipy import Dictionary, SplitMode
import kenlm
from datasets import load_dataset

doc_num = 100000

def read_text(data, ppl):
        if data["meta"]["ppl"] <= ppl:
            return data["text"]
        else:
            return ""

num = re.compile('[0-9]+')
symbol = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')
alpha = p = re.compile('[a-z]|[A-Z]')
dict = Dictionary()
tokenizer = dict.create(mode=SplitMode.C)

# 名詞を抽出, 分かち書き
def analyzer(text):
  words = []
  wakati_text = []
  for t in text.split("\n"):
    for s in t.split("。"):
      ms = tokenizer.tokenize(s)
      for m in ms:
        word = m.surface()
        wakati_text.append(word)
        if m.part_of_speech()[0] == "名詞":
          if num.fullmatch(word):
            words.append("0")
          elif symbol.fullmatch(word) or alpha.fullmatch(word):
            pass
          else:
            words.append(word)
  wakati_text = " ".join(wakati_text)
  return words, wakati_text

import requests

# ストップワードを取得
def get_stop_words():
    url = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
    r = requests.get(url)
    tmp = r.text.split('\r\n')
    stopwords_ja = []
    for i in range(len(tmp)):
        if len(tmp[i]) < 1:
            continue
        stopwords_ja.append(tmp[i])
    stopwords_ja += ["０","１","２","３","４","５","６","７","８","９"]
    return stopwords_ja

def get_topic_number(corpus, lda_model):
  topics_number = []
  for topics_per_document in lda_model[corpus]:
    topic_number = 0
    max_score = 0.0
    for topics in topics_per_document:
      if max_score < topics[1]:
        topic_number = topics[0]
        max_score = topics[1]
    topics_number.append(topic_number+1)
  return topics_number

#sudachi-kenlm
m = kenlm.LanguageModel("/scratch/ace14333cp/ja_llm/classifier/data/kenlm/model/kenlm_merge-code_0.05_model.bin")

def cal_ppl(wakati_text):
   ppl = m.perplexity(wakati_text)
   return ppl

def remove_meta_from_cc(text):
    paragraphs = text.split("\n\n")
    pars = []
    for par in paragraphs:
        idx = par.find('\x1c')
        if idx >= 0:
            par = par[idx + 1:]
        par = par.replace("\x02", "").replace("\x03", "")
        pars.append(par)
    return "\n\n".join(pars)

def read_cc(file):
    dataset = load_dataset("parquet", data_files=file)
    train_ds = dataset["train"]
    new_dataset = []
    for data in train_ds:
        new_data = {}
        new_data["text"] = remove_meta_from_cc(data["text"])
        new_data["meta"] = {"docId":data["docId"], "url":data["docId"], "charset":data["charset"], "date":data["date"]}
        new_dataset.append(new_data)
    return new_dataset

# 最初からトピックモデル学習
def train_topic_model():
    input_dir_path = "/scratch/ace14333cp/ja_llm/llm-jp-corpus/data/cc/ja_cc/raw"
    #dataset = "/scratch/ace14333cp/ja_llm/classifier/data/evaluate/mc4-ja_token-num/dir1/0000_mc4-ja_token-num.jsonl"
    target_ppls = [660, 4120, 57000, float('inf')]
    count = 0
    texts = []
    new_dataset = []
    print("start!")

    for i in range(0,100):
        file_index = str(i*10).zfill(5)
        input_file = os.path.join(input_dir_path, "part-" + file_index + "-55624510-b8c5-443c-8ca9-04d95450cbb6.c000.zstd.parquet")
        dataset = read_cc(input_file)
        for data in dataset:
            #clean text
            if count%1000 == 0:
                print("noun extract", count, "clear!")
            text = data["text"]
            if text != "":
                try:
                    words, wakati_text = analyzer(text)
                    texts.append(words)
                    ppl = cal_ppl(wakati_text)
                    count += 1
                    data["ppl"] = ppl
                    new_dataset.append(data)
                except:
                    pass
            if count >= doc_num:
                break
        else:
            continue
        break
    print("count:", count)

    en = re.compile('[a-z]+')
    stop_word = get_stop_words()
    tmp_texts = texts
    texts = [[w.lower() for w in p if (w.lower() not in stop_word) and not(en.fullmatch(w.lower()))] for p in tmp_texts]
    
    #save texts
    with open("topic_texts.jsonl", "w") as f:
        for nouns, data in zip(texts, new_dataset):
            data["nouns"] = nouns
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.save_as_text("dictionary.txt")
    dictionary.filter_extremes(no_below=3, no_above=0.3)
    corpus = [dictionary.doc2bow(t) for t in texts]
    num_topics = 17
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=0)
    vis_pcoa = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
    lda_model.save('lda_100k.model')
    pyLDAvis.save_html(vis_pcoa, 'pyldavis_lda_100k.html')

# 辞書からトピックモデル学習
def middle():
    file = "/scratch/ace14333cp/ja_llm/bert_train/topic_texts.jsonl"
    texts = []
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["nouns"])
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.save_as_text("dictionary.txt")
    dictionary.filter_extremes(no_below=3, no_above=0.3)
    corpus = [dictionary.doc2bow(t) for t in texts]
    print("created corpus!")
    num_topics = 16
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=0)
    vis_pcoa = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
    lda_model.save('lda_100k_tmp.model')
    print("saved lda model")
    pyLDAvis.save_html(vis_pcoa, 'pyldavis_lda_100k_tmp.html')

# 推論
def cal_topic_number():
    file = "/scratch/ace14333cp/ja_llm/bert_train/topic_texts.jsonl"
    texts = []
    dataset = []
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
            texts.append(data["nouns"])
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.save_as_text("dictionary.txt")
    dictionary.filter_extremes(no_below=3, no_above=0.3)
    corpus = [dictionary.doc2bow(t) for t in texts]
    print("created corpus!")
    topics_number = []
    lda_model = gensim.models.ldamodel.LdaModel.load("/scratch/ace14333cp/ja_llm/bert_train/lda_model/lda_100k_tmp.model")
    for topics_per_document in lda_model[corpus]:
        topic_number = 0
        max_score = 0.0
        for topics in topics_per_document:
            if max_score < topics[1]:
                topic_number = topics[0]
                max_score = topics[1]
        topics_number.append(topic_number+1)
    
    with open("/scratch/ace14333cp/ja_llm/bert_train/topic_ana_100k.jsonl", "w") as f:
        for data, topic in zip(dataset, topics_number):
            data["topic"] = topic
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

cal_topic_number()



