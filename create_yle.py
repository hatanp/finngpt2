

import os
import json
import sys
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from transformers import PreTrainedTokenizerFast
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel

from transformers import GPT2Model
from transformers import GPT2TokenizerFast
import transformers
import torch
import numpy as np

root = r'G:\Data\yle\data'#download from kielipankki and extract

texts = []
subjects = []
first_subjects = []
first_ids = []
subject_ids = []

for path, subdirs, files in os.walk(root):
    #Data is split into multiple files
    for name in files:
        print(os.path.join(path, name))
        with open(os.path.join(path, name), encoding="utf8") as f:
            data = json.load(f)

            #Each file contains json with multiple articles
            for i in range(len(data["data"])):
                try:
                    txt = ""
                    s = [] #Subjects
                    s_ids = []#Id for the subjects
                    #From the content loop trough the content and get only heading as text as we do not want to add metadata to a text dataset
                    for c in data["data"][i]["content"]:
                        if c["type"] in ("heading","text"):
                            txt += c["text"]
                        txt += "\n"
                    first = ""
                    #An article contains n subjects. Loop trough those and also save which one was first. We want that as a distinct column in the dataset for performance.
                    if "subjects" in data["data"][i]:#To know if we have a first subject, check first if we even have subjects in json.
                        first = data["data"][i]["subjects"][0]["title"]["fi"]
                        first_id = data["data"][i]["subjects"][0]["id"]
                        for subject in data["data"][i]["subjects"]:
                            s.append(subject["title"]["fi"])
                            s_ids.append(subject["id"])
                    first_subjects.append(first)
                    first_ids.append(first)
                    texts.append(txt)
                    subjects.append(s)
                    subject_ids.append(s_ids)
                except:
                    #Some texts contain formatting errors, just skip those as they are a neglible portion of all the articles.
                    pass


dataset = Dataset.from_dict({"text":texts, "subjects":subjects, "first_subject":first_subjects, "first_ids":first_ids, "subject_ids":subject_ids})

tokenizer_loc = "/media/volume/output/checkpoint-275000"

#output_dir = "/media/volume/fi_nlp/output/finetune"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_loc)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

def find_major_subject(example):
    good_subjects = ["urheilu","Kotimaan uutiset","Ulkomaat","jääkiekko","talous","politiikka","poliisi","Liikenne ja kuljetus","kulttuuri","puolueet","onnettomuudet","musiikki","Koulutus ja kasvatus","Venäjä","tieliikenne","luonto","autot","terveys","Helsinki","Pohjoismaat","kunnat","Eurooppa","rikokset","vaalit","Yhdysvallat","lainvalvonta"]
    import numpy as np
    example["main_subject"] = None
    label = np.zeros(len(good_subjects))
    for subject in example["subjects"]:
        if subject in good_subjects:
            example["main_subject"] = subject
            label[good_subjects.index(subject)] = 1
            #example["labels"] = label
            break
    return {"labels":label}

filtered = dataset.map(find_major_subject, num_proc=12).filter(lambda example: example['main_subject'] != None)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=800)
tokenized_and_filtered_dataset = filtered.map(tokenize_function, batched=True)

tokenized_and_filtered_dataset.save_to_disk(r"G:\Data\yle\tokenized_and_filtered_dataset")