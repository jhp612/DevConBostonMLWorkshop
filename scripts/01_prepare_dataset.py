import os
from urllib.request import urlopen
import json
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from tqdm import tqdm
from datasets import Dataset
import sys
import re

# If the parsed_poems.txt file doesn't exist, download and parse the source files
if not os.path.exists("parsed_poems.txt"):
    poems = []
    try:
        # Parse haiku text file
        haiku_url = "https://raw.githubusercontent.com/herval/creative_machines/master/haikuzao/src/main/resources/haiku.txt"
        print(f"Downloading and processing {haiku_url}")
        h_conn = urlopen(haiku_url)
        if h_conn.code == 200:
            rawtext = h_conn.read().decode()
            haikus = rawtext.split("\n\n")
            for h in haikus:
                h = h.replace(".","").replace("-","").replace("—","").replace("  "," ")

                tmp = h.split("\n")
                tmp = [t.strip() for t in tmp]
                if len(tmp) == 3:
                    poems.append(" / ".join(tmp))
                    #print(" / ".join(tmp))
        else:
            raise Exception(f"Failed to download {haiku_url}")

        # Parse unim_poem JSON file
        unim_poem_url = "https://raw.githubusercontent.com/researchmm/img2poem/master/data/unim_poem.json"
        print(f"Downloading and processing {unim_poem_url}")
        h_conn2 = urlopen(unim_poem_url)
        if h_conn2.code == 200:
            rawjson = h_conn2.read().decode()
            jrecs = json.loads(rawjson)
            for j in jrecs:
                tmp = j['poem'].split("\n")
                tmp = [t.strip() for t in tmp]
                if len(tmp) == 3:
                    poems.append(" / ".join(tmp))
                    #print(" / ".join(tmp))
        else:
            raise Exception(f"Failed to download {unim_peom_url}")

        with open("parsed_poems.txt","w") as f:
            for p in poems:
                f.write(p + "\n")

        print(f"=> Saved {len(poems)} poems to parsed_poems.txt")

    except Exception as e:
        print(e)
        print("\nPlease try again. If the error persists, please see your workshop facilitator")


# Now create a dataset from the parsed_poems.txt file
print("\nCreating new dataset from parsed poems file")
MAX_LEN=64  # file will be tokenized into sequences of MAX_LEN tokens

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
input_ids = []
attention_mask = []
lines_parsed = []

with open("parsed_poems.txt","r") as f:
    lines_parsed = f.readlines()

NUM_POEMS = len(lines_parsed)
pbar = tqdm(total=NUM_POEMS)

COUNT=30
# concatenate several poems per line to make sequences longer 
for n in range(NUM_POEMS//COUNT):
    input_seq = "".join(lines_parsed[COUNT*n:COUNT*n+COUNT])
    tokenized = tokenizer(input_seq, padding='max_length', truncation=True, return_tensors='pt', max_length=MAX_LEN)
    input_ids.extend(tokenized['input_ids'])
    attention_mask.extend(tokenized['attention_mask'])
    pbar.update(COUNT)

ds_dict = { "input_ids": input_ids, "attention_mask": attention_mask }

# build new dataset from dict
ds = Dataset.from_dict(ds_dict).with_format("torch")
print(f"dataset: {ds}")

# save dataset
ds.save_to_disk(f"poems_dataset_{MAX_LEN}")
