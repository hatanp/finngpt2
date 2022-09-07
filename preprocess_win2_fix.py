from datasets import load_dataset, concatenate_datasets
import transformers
import datasets
from transformers import (
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)

from datasets import Dataset, load_dataset
import os

input_dir = "H:\\Data_temp\\step_data" #Output from a part 1 script
tokenizer_file=r"C:\Users\vin\Documents\Projects\NLP\models\from_gpt2"
output_dir="H:\\Data_temp\\1024_bpe_dataset4"
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_file)

from tokenizers.processors import TemplateProcessing
#Required for adding the EOS tokens. Vanilla GPT-2 tokenizer does not add those.
tokenizer._tokenizer.post_processor = TemplateProcessing(
    single="$0 "+tokenizer.eos_token,
    pair="$A "+tokenizer.eos_token+" $B:1 "+tokenizer.eos_token,
    special_tokens=[(tokenizer.eos_token, 0)],
)

def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    """
    Group text into blocks of 1024 (context size) for efficient training. Copied from huggingface CLM example. 
    """
    block_size = 1024
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = len(concatenated_examples[list(examples.keys())[0]])
    total_len = (total_len//block_size) * block_size
    result = {
        k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    num_proc=12
    dataset = datasets.load_from_disk(input_dir)#Custom data
    yle_set = datasets.load_from_disk(r"G:\Data\yle\yle_set").remove_columns(["first_subject","subjects","first_ids","subject_ids"])#Just text from yle set
    mc4 = load_dataset("Finnish-NLP/mc4_fi_cleaned", split="train").remove_columns(["timestamp","url"])#Data from hf hub
    dataset = concatenate_datasets([dataset,yle_set, mc4])
    #Shuffle ensures even load for processes as mc4 was much larger. writer_batch_size of 100000 was not found to run out of memory. Disable cache to not accidentally load old data.
    dataset\
        .shuffle(seed=42, load_from_cache_file=False, writer_batch_size=100000)\
        .map(tokenize_function, batched=True, num_proc=num_proc, remove_columns=dataset.column_names, load_from_cache_file=False, writer_batch_size=100000)\
        .filter(lambda e: len(e["input_ids"]) > 20, num_proc=num_proc, load_from_cache_file=False, writer_batch_size=100000)\
        .map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=False, writer_batch_size=100000)\
        .train_test_split(test_size=0.05, load_from_cache_file=False, writer_batch_size=100000)\
        .save_to_disk(output_dir)
    print("done")

if __name__ == "__main__":
    main()
