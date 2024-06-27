from tqdm import tqdm
import json
import gensim
import fire



def cal_topic_number(work_dir, jsonl_input_file, jsonl_output_file, lang):

    dictionary = gensim.corpora.Dictionary()
    dictionary.load_from_text(f"{work_dir}/dictionary.filtered.txt")
    # dictionary.filter_extremes(no_below=3, no_above=0.3)
    topics_number = []
    lda_model = gensim.models.ldamodel.LdaModel.load(f"{work_dir}/lda_100k.model")

    with open(jsonl_output_file, "w") as fw:
        with open(jsonl_input_file, "r") as f:
            for line in tqdm(f):
                data = json.loads(line)
                if lang == "jp":
                    text = data["nouns"]
                elif lang == "en":
                    text = data["text"].lower().split()
                else:
                    raise ValueError("lang must be 'jp' or 'en'")

                corpus = dictionary.doc2bow(text)
                topics_per_document = lda_model[corpus][0]
                topic_number = 0
                max_score = 0.0
                import ipdb; ipdb.set_trace()
                for topics in topics_per_document:
                    if max_score < topics[1]:
                        topic_number = topics[0]
                        max_score = topics[1]
        
                data["topic"] = topic_number + 1
                json.dump(data, fw, ensure_ascii=False)
                fw.write("\n")

if __name__ == "__main__":
    # work_dir='output.en.17'
    # cal_topic_number (work_dir, "corpus/en_pile_merge_2.200k.jsonl", lang="en")
    fire.Fire(cal_topic_number)