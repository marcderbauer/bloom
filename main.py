from datasets import load_dataset
from transformers import BloomTokenizerFast, BloomForCausalLM, Trainer, TrainingArguments
import argparse
"""
References:
https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
https://huggingface.co/docs/transformers/model_doc/bloom
"""

# Kept these variables seperate, as modifying them could break training
BLOCK_SIZE = 128
PROCESSES = 4

#----------------------------------------------------------------------------
#                               ARGPARSE
#----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Main file for finetuning BLOOM')
parser.add_argument("--checkpoint", type=str, default="bigscience/bloom-560m", help="huggingface checkpoint to load model from")
parser.add_argument("--train", type=str, default="data/train.txt", help="path to train.txt")
parser.add_argument("--test", type=str, default="data/test.txt", help="path to test.txt")
parser.add_argument("--output_dir", type=str, default=None, help="Folder to save final model into")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--wd", type=float, default=0.01, help="Weight Decay")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=1000, help="Samples per batch")
parser.add_argument("--wandb", default=None, action='store_const', const="wandb")

args = parser.parse_args()

model_name = args.checkpoint.split("/")[-1]
if not args.output_dir:
    output_dir = f"{model_name}-vice-headlines"

tokenizer = BloomTokenizerFast.from_pretrained(args.checkpoint)
dataset = load_dataset("text", data_files={"train": args.train, "test": args.test})
model = BloomForCausalLM.from_pretrained(args.checkpoint)


def tokenize_function(examples):
    examples["text"] = [example + " </s>" for example in examples["text"]] # Append EOS
    tokenized = tokenizer(examples["text"], padding=True, pad_to_multiple_of=8)
    return tokenized

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Drop the small remainder
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    # Process dataset
    tokenized = dataset.map(tokenize_function, batched=True, num_proc=PROCESSES, remove_columns=["text"])
    lm_datasets = tokenized.map(group_texts, batched=True, batch_size=args.batch_size, num_proc=PROCESSES,)

    training_args = TrainingArguments(
        output_dir,
        evaluation_strategy = "epoch",
        learning_rate=args.lr,
        weight_decay=args.wd,
        num_train_epochs=args.epochs,
        report_to=args.wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"]
)
    print("Starting training")
    trainer.train()
    print("Finished training")
    model.save_pretrained(output_dir)
    print(trainer)

if __name__ == "__main__":
    main()