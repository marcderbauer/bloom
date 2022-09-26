from transformers import BloomTokenizerFast, BloomForCausalLM
import torch as pt
import argparse
import os
from titlecase import titlecase

#----------------------------------------------------------------------------
#                               ARGPARSE
#----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate inference for a given prompt')
parser.add_argument("prompt", type=str, nargs="+", help="The prompt to generate inference from.")
parser.add_argument("--checkpoint", type=str, default="bigscience/bloom-560m", help="huggingface checkpoint to load model from")
parser.add_argument("--temp", type=float, default=0.7, help="Temperature for inference (0-1). Lower is more chaotic.")
parser.add_argument("--top_p", type=float, default=0.92)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--rp", type=float, default=1.2, help="Repetition penalty")
parser.add_argument("--max_length", type=int, default=30)
parser.add_argument("--min_length", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

if args.min_length == None:
    args.min_length = len(args.prompt) + 1
if args.seed:
    pt.manual_seed(args.seed)
assert args.max_length > args.min_length


def all_subdirs_of(b='.'):
    """
    
    https://stackoverflow.com/questions/2014554/find-the-newest-folder-in-a-directory-in-python
    """
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd): result.append(bd)
    return result

#----------------------------------------------------------------------------
#                               MODEL
#----------------------------------------------------------------------------
# TODO: Wouldn't it make more sense to have the model dir as argument directly instead of the checkpoint?
#   Would allow the user to choose best checkpoint in case of overfitting etc.
model_name = args.checkpoint.split("/")[-1]
model_path_all = f"{model_name}-vice-headlines"
latest_model = max(all_subdirs_of(model_path_all), key=os.path.getmtime) # Use the most_recent_model
tokenizer = BloomTokenizerFast.from_pretrained(args.checkpoint)
model = BloomForCausalLM.from_pretrained(latest_model)

#----------------------------------------------------------------------------
#                               INFERENCE
#----------------------------------------------------------------------------

# generate
input_ids = tokenizer.encode(" ".join(args.prompt), return_tensors='pt')

# Top-p sampling
# More info here: https://huggingface.co/blog/how-to-generate
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=args.max_length, 
    min_length=args.min_length,
    top_p=args.top_p, 
    top_k=args.top_k,
    temperature=args.temp,
    repetition_penalty=args.rp
)

print(f"\ntemp={args.temp}; k={args.top_k}, p={args.top_p}, rep={args.rp}\n" + 100 * '-')
print(titlecase(tokenizer.decode(sample_output[0], skip_special_tokens=True)))