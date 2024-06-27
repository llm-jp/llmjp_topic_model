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
import kenlm
from datasets import load_dataset
import glob
from datetime import datetime
import fire
import utils

doc_num = 100000

def read_text(data, ppl):
        if data["meta"]["ppl"] <= ppl:
            return data["text"]
        else:
            return ""

num = re.compile('[0-9]+')
symbol = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')
alpha = p = re.compile('[a-z]|[A-Z]')

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

def load_text_ja(calc_ppl=False):
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
                    texts.append(text)
                    if calc_ppl:
                        words, wakati_text = analyzer(text)
                        ppl = cal_ppl(wakati_text)
                        data["ppl"] = ppl
                    count += 1
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
    texts = [d['text'] for d in dataset]
    # import ipdb; ipdb.set_trace()
    print("texts:", len(texts))
    return texts, dataset

# 最初からトピックモデル学習
def train_topic_model(work_dir, texts, new_dataset, lang="jp", num_topics=17):
    tokenizer = utils.Tokenizer(lang)
    texts = [tokenizer(p) for p in tqdm(texts, desc="Tokenizing")]
    
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
    print("saved lda model")

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
    tokenizer = utils.Tokenizer(lang)
    # file = f"{work_dir}/topic_texts.jsonl"
    texts = []
    dataset = []
    with open(file, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            dataset.append(data)
            texts.append(tokenizer(data["text"]))
    # dictionary = gensim.corpora.Dictionary(texts)
    # dictionary.save_as_text(f"{work_dir}/dictionary.txt")
    # dictionary.filter_extremes(no_below=3, no_above=0.3)
    dictionary = gensim.corpora.Dictionary.load(f"{work_dir}/dictionary.filtered.pkl")
    # import ipdb; ipdb.set_trace()
    corpus = [dictionary.doc2bow(t) for t in texts]
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
        train_topic_model(work_dir, texts, new_dataset, lang='jp', num_topics=num_topics)
        cal_topic_number (work_dir, "corpus/CC-MAIN.txt", lang="jp")
    elif lang=="en":
        texts, new_dataset = load_text_en()
        train_topic_model(work_dir, texts, new_dataset, lang='en', num_topics=num_topics)
        cal_topic_number (work_dir, "corpus/en_pile_merge_2.200k.jsonl", lang="en")

if __name__ == "__main__":
    fire.Fire(main)
