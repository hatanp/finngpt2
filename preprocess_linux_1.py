from datasets import load_dataset, concatenate_datasets
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)

from datasets import Dataset
import os


input_dir = "/mnt/h/Data_temp/ready_data" #A folder with .txt files where chapters are separated by EOT_token
output_dir="/mnt/h/Data_temp/step_data"

EOT_token = "<|endoftext|> "

count = 0
texts = []
dataset = None
"""
A text chapter might contain newlines and we want to add all the lines to a dataset entry. Thus using newline as a text separator does not work so we have a EOT_token to represent change of context.
"""
filelist = [fil.path for fil in os.scandir(input_dir)]
for filename in filelist:
    with open(filename, "r", encoding="utf-8") as in_f:
        item = ""
        for line in in_f:
            if EOT_token in line:
                parts = line.split(EOT_token)#We have multiple entries on a single line.
                texts.append(item+parts[0])
                #As a compromise between throughput and memory usage, make a new dataset object from the dict every n texts. This is inside the EOT_token check on purpose as this is not that important
                if len(texts) > 20000000:
                    temp_set = Dataset.from_dict({"text":texts})
                    if dataset == None:#If a dataset object does not exist, make the current one the master
                        dataset = temp_set
                    else:#Add to an existing master dataset
                        dataset = concatenate_datasets([dataset, temp_set])
                    texts = []
                    count += 1
                    print(count)
                #Add all the complete parts to separate entries. Last one continues so don't add that as such. 
                for part in parts[1:-1]:
                    texts.append(part)
                item = parts[-1]
            else: #we are inside a text chapter, just keep going.
                item += line
#Add the leftovers if we have any
if len(texts) > 0:
    temp_set = Dataset.from_dict({"text":texts})
    dataset = concatenate_datasets([dataset, temp_set])

dataset.save_to_disk(output_dir)