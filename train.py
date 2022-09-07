
import transformers
import datasets
from transformers import PreTrainedTokenizerFast
from transformers import (
    GPT2TokenizerFast,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint
import torch
#from transformers.utils.dummy_tokenizers_objects import PreTrainedTokenizerFast

def main():
    import os
    #Only use my rtx 3090 and to prevent trying to use data parallel with my gt 1030
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #Not sure if required
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    config_name = "config.json"
    tokenizer_files = "C:\\Users\\vin\\Documents\\Projects\\NLP\\models\\from_gpt2"
    input_dir = "H:\\Data_temp\\1024_bpe_dataset4"
    output_dir = "H:\\Data_temp\\checkpoints\\bpe_large"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,#Max to fit into VRAM
        per_device_eval_batch_size=4,
        learning_rate=2.067e-5,#Initial of 3e-5 with a few epochs done
        lr_scheduler_type="linear",
        adam_beta1=0.95,
        adam_beta2=0.985,#Found to work well with grid search
        adam_epsilon=1e-8,
        weight_decay=0.001,
        gradient_accumulation_steps=32,#To get a reasonable practical batch size of 128. Also helps with throughput as gradients are applied less often
        num_train_epochs=6.7,
        save_total_limit=2,
        dataloader_num_workers=10,
        save_steps=100,
        warmup_steps=1000,
        do_eval=True,
        eval_steps=1000,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=100,
        bf16=True,
        tf32=True,
        fp16_opt_level="O2",
        half_precision_backend="amp",
        bf16_full_eval=True
    )

    print("setting up tokenizer...")
    #tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_files)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    from tokenizers.processors import TemplateProcessing
    #Required for adding the EOS tokens. Vanilla GPT-2 tokenizer does not add those.
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="$0 "+tokenizer.eos_token,
        pair="$A "+tokenizer.eos_token+" $B:1 "+tokenizer.eos_token,
        special_tokens=[(tokenizer.eos_token, 0)],
    )

    print("loading model...")
    config = AutoConfig.from_pretrained(config_name)
    model = AutoModelForCausalLM.from_config(config)
    #uncomment if loading only weights from some checkpoint, not to be confused with full checkpoint that contains also scheduler state
    #model = AutoModelForCausalLM.from_pretrained(r"H:\Data_temp\checkpoints\good_large\checkpoint-95400")
    
    #Could affect performance
    model.gradient_checkpointing_enable()

    print("loading data...")
    dataset = datasets.load_from_disk(input_dir)

    print("starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"].select(range(10000)),
        data_collator=default_data_collator,
        tokenizer=tokenizer
    )
    #Resume training from checkpoint fully including scheduler state
    #checkpoint = None
    checkpoint = get_last_checkpoint(output_dir)
    print("checkpoint:", checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)

if __name__ == "__main__":
    main()