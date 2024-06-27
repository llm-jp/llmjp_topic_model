import re
import numpy as np
from tqdm import tqdm
# import math
# import MeCab
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
import glob
from datetime import datetime
import fire

doc_num = 100000

def read_text(data, ppl):
        if data["meta"]["ppl"] <= ppl:
            return data["text"]
        else:
            return ""

num = re.compile('[0-9]+')
symbol = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')
alpha = p = re.compile('[a-z]|[A-Z]')
# dict = Dictionary()
tokenizer = Dictionary().create(mode=SplitMode.C)

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
def get_stop_words(lang="jp"):
    if lang=="jp":
        url = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
    elif lang=="en":
        url = "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
    else:
        raise
    r = requests.get(url)
    tmp = r.text.split('\r\n')
    if lang=="jp":
        stopwords = []
        for i in range(len(tmp)):
            if len(tmp[i]) < 1:
                continue
            stopwords.append(tmp[i])
        stopwords += ["０","１","２","３","４","５","６","７","８","９"]
    else:
        stopwords = tmp[0].split('\n')
    return stopwords

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
m = kenlm.LanguageModel("kenlm_merge-code_0.05_model.bin")

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
    # dataset = load_dataset("parquet", data_files=file)
    print(file)
    dataset = load_dataset("json", data_files=file)
    train_ds = dataset["train"]
    new_dataset = []
    for data in train_ds:
        new_data = {}
        new_data["text"] = remove_meta_from_cc(data["text"])
        if "docId" in data:
            new_data["meta"] = {"docId":data["docId"], "url":data["docId"], "charset":data["charset"], "date":data["date"]}
        new_dataset.append(new_data)
    return new_dataset

# Textize datetime obj before JSONize
def serialize_dates(obj):
    if isinstance(obj, dict):
        return {k: serialize_dates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_dates(i) for i in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

def load_text_ja():
    # input_dir_path = "/scratch/ace14333cp/ja_llm/llm-jp-corpus/data/cc/ja_cc/raw"
    #dataset = "/scratch/ace14333cp/ja_llm/classifier/data/evaluate/mc4-ja_token-num/dir1/0000_mc4-ja_token-num.jsonl"
    target_ppls = [660, 4120, 57000, float('inf')]
    count = 0
    texts = []
    new_dataset = []
    folder_path = "/data/llm-jp-corpus-v2.1-CC/filtered_v3.0/segment=*/filter=null/"
    paths = sorted(glob.glob(folder_path))

    for i in range(0,100):
        # file_index = str(i*10).zfill(5)
        # input_file = os.path.join(input_dir_path, "part-" + file_index + "-55624510-b8c5-443c-8ca9-04d95450cbb6.c000.zstd.parquet")
        path = paths[i%len(paths)]
        gz_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.gz')])
        dataset = read_cc(gz_files[i%len(paths)])
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
    print("count:", count, "texts:", len(texts))
    return texts, new_dataset

def load_text_en():
    path = "corpus/en_pile_merge_1.200k.jsonl"
    dataset = read_cc(path)
    texts = [d['text'].lower().split() for d in dataset]
    # import ipdb; ipdb.set_trace()
    print("texts:", len(texts))
    return texts, dataset

class Tokenizer:
    def __init__(self,lang):
        self.stop_word = get_stop_words(lang)
        self.lang = lang
        self. en = re.compile('[a-z]+')
    def __call__(self, text):
        if self.lang == "jp":
            words, wakati = analyzer(text)
            return [w.lower() for w in words if (w.lower() not in self.stop_word) and not(self.en.fullmatch(w.lower()))]
        elif self.lang == "en":
            return [w.lower() for w in text.split() if (w.lower() not in self.stop_word)]
        else:
            raise ValueError("lang must be 'jp' or 'en'")

# 最初からトピックモデル学習
def train_topic_model(work_dir, texts, new_dataset, lang="jp", num_topics=17):
    tokenizer = Tokenizer(lang)
    texts = [tokenizer(p) for p in texts]
    
    #save texts
    with open(f"{work_dir}/topic_texts.jsonl", "w") as f:
        for nouns, data in zip(texts, new_dataset):
            data["nouns"] = nouns
            json.dump(serialize_dates(data), f, ensure_ascii=False)
            f.write("\n")
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.save_as_text(f"{work_dir}/dictionary.txt")
    dictionary.filter_extremes(no_below=3, no_above=0.3)
    dictionary.save(f"{work_dir}/dictionary.filtered.pkl")
    corpus = [dictionary.doc2bow(t) for t in texts]
    # import ipdb; ipdb.set_trace()

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=0)
    vis_pcoa = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
    lda_model.save(f'{work_dir}/lda_100k.model')
    pyLDAvis.save_html(vis_pcoa, f'{work_dir}/pyldavis_lda_100k.html')

# 辞書からトピックモデル学習
def middle(work_dir):
    file = "/scratch/ace14333cp/ja_llm/bert_train/topic_texts.jsonl"
    texts = []
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["nouns"])
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.save_as_text(f"{work_dir}/dictionary.txt")
    dictionary.filter_extremes(no_below=3, no_above=0.3)
    corpus = [dictionary.doc2bow(t) for t in texts]
    print("created corpus!")
    num_topics = 16
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=0)
    vis_pcoa = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
    lda_model.save('lda_100k.model')
    print("saved lda model")
    pyLDAvis.save_html(vis_pcoa, 'pyldavis_lda_100k.html')

# 推論
def cal_topic_number(work_dir,file, lang="jp"):
    # file = f"{work_dir}/topic_texts.jsonl"
    texts = []
    dataset = []
    if lang == "jp":
        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                dataset.append(data)
                texts.append(data["nouns"])
    elif lang == "en":
        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                dataset.append(data)
                texts.append(data["text"].lower().split())
    else:
        raise ValueError("lang must be 'jp' or 'en'")
    dictionary = gensim.corpora.Dictionary(texts)
    # dictionary.save_as_text(f"{work_dir}/dictionary.txt")
    dictionary.filter_extremes(no_below=3, no_above=0.3)
    corpus = [dictionary.doc2bow(t) for t in texts]
    print("created corpus!")
    topics_number = []
    lda_model = gensim.models.ldamodel.LdaModel.load(f"{work_dir}/lda_100k.model")
    for topics_per_document in lda_model[corpus]:
        topic_number = 0
        max_score = 0.0
        for topics in topics_per_document:
            if max_score < topics[1]:
                topic_number = topics[0]
                max_score = topics[1]
        topics_number.append(topic_number+1)
    
    with open(f"{work_dir}/topic_ana_100k.jsonl", "w") as f:
        for data, topic in zip(dataset, topics_number):
            data["topic"] = topic
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

def main(lang, num_topics=17):

    work_dir=f'output.{lang}.{num_topics}'
    os.makedirs(work_dir, exist_ok=True) 

    if lang=="jp":
        texts, new_dataset = load_text_ja()
        train_topic_model(work_dir, texts, new_dataset, lang='jp', num_topics=17)
        cal_topic_number (work_dir, "corpus/CC-MAIN.txt", lang="jp")
    # texts, new_dataset = load_text_ja()
    # train_topic_model(work_dir, texts, new_dataset, lang='jp', num_topics=17)
    # cal_topic_number (work_dir)
    elif lang=="en":
        texts, new_dataset = load_text_en()
        train_topic_model(work_dir, texts, new_dataset, lang='en', num_topics=17)
        cal_topic_number (work_dir, "corpus/en_pile_merge_2.200k.jsonl", lang="en")

if __name__ == "__main__":
    fire.Fire(main)
