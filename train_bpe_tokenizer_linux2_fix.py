from transformers import AutoTokenizer
import datasets
#Train based on GPT-2 tokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
#Training corpus
input_dir = "/mnt/h/Data_temp/step_data"
dataset = datasets.load_from_disk(input_dir)

#An iterator to get n samples from the training corpus
def get_training_corpus():
    for start_idx in range(0, len(dataset), 10000):
        samples = dataset[start_idx : start_idx + 10000]
        yield samples["text"]

print("start")
tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), vocab_size=50000)
print("end")
tokenizer.save_vocabulary("/mnt/c/Users/vin/Documents/Projects/NLP/models/from_gpt2")
