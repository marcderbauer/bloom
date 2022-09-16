from datasets import load_dataset, load_metric
from transformers import BloomTokenizerFast, BloomForCausalLM, Trainer, TrainingArguments
import torch
import numpy as np


"""
References:
https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
https://huggingface.co/docs/transformers/model_doc/bloom
"""

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
dataset = load_dataset("text", data_files={"train": "data/train.txt", "test": "data/test.txt"})
metric = load_metric("accuracy")


# Use MPS if available
#mps_device = torch.device("mps")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Tokenize texts and then groupt them according to block_size
# block_size = tokenizer.model_max_length -> 1000000000000000019884624838656 -- too much
block_size = 128
tokenized = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

lm_datasets = tokenized.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

print(f"\n\n{'-' * 100}\n\n")
print(tokenizer.decode(lm_datasets["train"][5]["input_ids"]))

#  Define model
model_checkpoint = "bigscience/bloom-560m"
model_name = model_checkpoint.split("/")[-1]
model = BloomForCausalLM.from_pretrained(model_checkpoint)
#model.to(mps_device)

training_args = TrainingArguments(
    f"{model_name}-vice-headlines",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"]
    #compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("models")
print(trainer)
