import os
import torch
from transformers import AutoModelForCausalLM, GPT2Config
from transformers import GPT2LMHeadModel
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.gpt2.model import GPT2ForSampling
from glob import glob
import re

model_name = 'gpt2-ft'
splitfile = f"./{model_name}-split"

checkpoints = glob("./output/checkpoint*")
cnums = []
for c in checkpoints:
    m = re.search(r"checkpoint-(\d+)", c)
    cnums.append(int(m.group(1)))
cnums.sort()
checkpoint = f"output/checkpoint-{cnums[-1]}"
print(f"Using checkpoint file {checkpoint}")

# Create the CPU model (only need to do once per model)
print('  creating the CPU model ...')
model_cpu = GPT2LMHeadModel.from_pretrained(checkpoint)

def amp_callback(model, dtype):
    # cast attention and mlp to low precisions only; layernorms stay as f32
    for block in model.transformer.h:
        block.attn.to(dtype)
        block.mlp.to(dtype)
    model.lm_head.to(dtype)

#print('converting the model to bf16 ...')
#amp_callback(model_cpu, torch.bfloat16)

print('  saving the model split ...')
save_pretrained_split(model_cpu, splitfile)
print(f"Saved {splitfile}")
