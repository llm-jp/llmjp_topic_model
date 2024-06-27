import os
import gzip
import json
import re
from tqdm import tqdm
import glob

#folder_path = "/data/llm-jp-corpus-v2.1-CC/filtered_v3.0/segment=CC-MAIN-2017-04/filter=null/"
folder_path = "/data/llm-jp-corpus-v2.1-CC/filtered_v3.0/segment=*/filter=null/"

# match = re.search(r'segment=(CC-MAIN-\d{4}-\d{2})', folder_path)
# if match:
#     segment = match.group(1)
# else:
#     raise ValueError("Not found")

output_file = f'CC-MAIN.txt'

with open(output_file, 'w', encoding='utf-8') as outfile:
    for path in tqdm(sorted(glob.glob(folder_path)), desc="Processing files"):
        print(path)
        gz_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.gz')])

        # for file in tqdm(gz_files, desc="Processing files"):
        for file in gz_files[:3]:
            for line in gzip.open(file, 'rt', encoding='utf-8'):
                data = json.loads(line)
                if 'text' in data:
                    text = data['text'].replace('\n', ' ').replace('\r', ' ')
                    outfile.write(f"{json.dumps({'text':text}, ensure_ascii=False)}\n")
            break
