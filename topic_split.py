import json
import fire

def split_topic_data(json_path):
    topic_data = {}

    for line in open(json_path, 'r', encoding='utf-8'):
        data = json.loads(line)
        topic = data["topic"]
        text = data["text"]
        if topic not in topic_data:
            topic_data[topic] = []
        topic_data[topic].append({"text": text})

    print("Stats of topic_data dictionary:")
    for topic, data_list in topic_data.items():
        print(f"Topic: {topic}, Count: {len(data_list)}")

    for topic, texts in topic_data.items():
        file_name = f"{json_path}.split.{topic}.jsonl"
        with open(file_name, 'w', encoding='utf-8') as f:
            for text_entry in texts:
                json.dump(text_entry, f, ensure_ascii=False)
                f.write('\n')

if __name__ == "__main__":
    work_dir='output7'
    split_topic_data(f"{work_dir}/topic_ana_100k.jsonl")