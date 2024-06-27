from tqdm import tqdm
import json
import gensim
import fire
from utils import topic_tokenizer

def cal_topic_number(work_dir, jsonl_input_file, jsonl_output_file, lang):

    tokenizer = topic_tokenizer.Tokenizer(lang)
    dictionary = gensim.corpora.Dictionary.load_from_text(f"{work_dir}/dictionary.filtered.txt")
    lda_model = gensim.models.ldamodel.LdaModel.load(f"{work_dir}/lda_100k.model")

    # total = 1e7 if lang == "en" else 1e6

    with open(jsonl_output_file, "w") as fw:
        with open(jsonl_input_file, "r") as f:
            for line in tqdm(f, total=1e7):
                data = json.loads(line)
                text = tokenizer(data["text"])

                corpus = dictionary.doc2bow(text)
                topics_per_document = lda_model[corpus]
                topic_number = 0
                max_score = 0.0
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