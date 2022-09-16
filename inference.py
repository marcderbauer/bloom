from transformers import BloomTokenizerFast, BloomForCausalLM
import torch as pt

# pt.manual_seed(0)

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLM.from_pretrained("models/")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs, labels=inputs["input_ids"])
# loss = outputs.loss
# logits = outputs.logits

# print(logits)


# generate
input_ids = tokenizer.encode('North Korean', return_tensors='pt')
"""

# activate beam search and early_stopping
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2,
    early_stopping=True
)

print("\nBeam output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))


# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0
)

print("\nSample output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


# adding temperature to the sample output
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0,
    temperature=0.7
)

print("\nSample output with Temperature = 0.7:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


# Top K sampling
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=50
)

print("\nTop k=50 sampling:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))



# Top-p sampling -- deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_p=0.92, 
    top_k=0
)

print("\nTop P sampling output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


# Top-p sampling -- deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_p=0.92, 
    top_k=0,
    temperature=0.7
)

print("\nTop with temp = 0.7:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

"""

# Top-p sampling -- deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=20, 
    top_p=0.92, 
    top_k=50,
    temperature=0.7,
    repetition_penalty=1.2
)

print("\nTop with temp=0.7; k=50, p=0.92,rep=1.2:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))