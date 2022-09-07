

import os
import sys
from datasets import load_dataset, load_from_disk, concatenate_datasets
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


def compute_metrics(eval_pred):
    """
    Calculate accuracy of all the labels being correct for an example
    """
    logits,labels = eval_pred
    #A poor way to make a boolean matrix
    predictions = np.zeros(logits.shape)
    #A bad way to index too...
    predictions[np.arange(len(predictions)),logits.argmax(1)] = 1
    predictions = predictions > 0.5

    #predictions = logits > 0.5
    #Boolean matrix for the collect labels
    labels = labels > 0.5
    return {"acc":np.all(predictions == labels,axis=1).sum()/predictions.shape[0]}

class MultilabelTrainer(Trainer):
    """
    Trainer with default loss function overloaded to allow for multi-label multi-class prediction
    """
    def compute_loss(self,model,inputs,return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1,self.model.config.num_labels),
        labels.float().view(-1,self.model.config.num_labels))
        return (loss,outputs) if return_outputs else loss

def main():

    print("cuda_avail:",torch.cuda.is_available())
    #checkpoint_loc = "/media/volume/output/checkpoint-275000"
    #output_dir = "/media/volume/fi_nlp/output/finetune"
    checkpoint_loc = r"H:\Data_temp\checkpoints\good_large\checkpoint-67400"#Model location to start finetuning from
    output_dir = r"H:\Data_temp\checkpoints\tests\yle"

    """training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-6,
        num_train_epochs=15, 
        save_total_limit=2,
        dataloader_num_workers=2,
        save_steps=10000,
        do_eval=True,
        eval_steps=500,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=500
    )"""

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=3e-7,
        adam_epsilon=1e-8,
        weight_decay=0.001,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=10,
        num_train_epochs=1,
        save_total_limit=2,
        dataloader_num_workers=5,
        save_steps=10000,
        warmup_steps=200,
        do_eval=True,
        eval_steps=200,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=50,
        no_cuda=True
    )

    print(training_args)

    dataset = load_from_disk(r"G:\Data\yle\tokenized_and_filtered_dataset")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_loc)
    num_labels = len(dataset[0]["labels"])
    print("num_labels",num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_loc,num_labels=num_labels)
    #model = AutoModelForSequenceClassification.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1",num_labels=9)

    #This can cause train-test leakage due to some huggingface bug. TODO: investigate
    split_datasets = dataset.train_test_split(test_size=0.1)
    print(split_datasets)
    print("init trainer")
    trainer = MultilabelTrainer(
            model=model,
            args=training_args,
            train_dataset=split_datasets["train"],
            eval_dataset=split_datasets["test"].select(range(1000)),#To not waste time running whole evaluation test every time
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=default_data_collator
        )
    checkpoint = None
    #checkpoint = get_last_checkpoint(output_dir)
    #checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #trainer.save_model()  # Saves the tokenizer too

if __name__ == "__main__":
    main()
