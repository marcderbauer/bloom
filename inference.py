from transformers import BloomTokenizerFast, BloomForCausalLM
import torch as pt
import argparse

#----------------------------------------------------------------------------
#                               ARGPARSE
#----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate inference for a given prompt')
parser.add_argument("prompt", type=str, nargs="+", help="The prompt to generate inference from.")
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

#----------------------------------------------------------------------------
#                               MODEL
#----------------------------------------------------------------------------

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLM.from_pretrained("upload") #TODO: make this one consistent with output dir of main

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

print(f"\ntemp={args.temp}; k={args.top_k}, p={args.top_p}, rep={args.rp}:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))