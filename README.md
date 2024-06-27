```sh
python topic_ana_predict.py en 17
python topic_ana_predict.py jp 17

python topic_predicter.py output.en.17 /data/llm-jp-corpus/v1.0.1/merge/en_pile/en_pile_merge_2.jsonl /model/llm-jp-corpus/v1.0.1/topic.17/en_pile/en_pile_merge_2.jsonl en
source venv310/bin/activate; file=CC-MAIN-2017-04.jsonl; python topic_predicter.py output.jp.17 /data/llm-jp-corpus-v2.1-CC/merge/$file /model/llm-jp-corpus/v1.0.1/topic.17/ja_CC/$file jp

python topic_split.py output.en.17
python topic_split.py output.jp.17
```