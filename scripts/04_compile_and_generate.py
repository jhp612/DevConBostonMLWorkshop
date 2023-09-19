import os
import torch
import re
import time
import argparse
from transformers import AutoModelForCausalLM, GPT2Config, AutoTokenizer, GPT2LMHeadModel
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.gpt2.model import GPT2ForSampling
from transformers_neuronx.config import NeuronConfig, QuantizationConfig

os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'
parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--seqlen", type=int, default=64)
parser.add_argument("--tp", type=int, default=2)
args = parser.parse_args()

model_name = 'gpt2-ft'

print('Creating the Neuron model ...')
model_neuron = GPT2ForSampling.from_pretrained(
    f'./{model_name}-split', 
    batch_size=args.bs,
    tp_degree=args.tp,
    n_positions=args.seqlen,
    amp='f32', 
)
print(f"Compiling the Neuron model for batch_size={args.bs}, sequence_length={args.seqlen}, tp_degree={args.tp} ...")
model_neuron.to_neuron()

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id

def sample(model_neuron, text, seqlen):
    encoded_input = tokenizer(text, return_tensors='pt', padding="longest")
    input_ids = encoded_input.input_ids
    input_length = input_ids.shape[1]
    new_tokens = seqlen - input_length

    print(f'input prompt length: {input_length}')
    print(f'generated tokens: {new_tokens}')

    start = time.time()
    model_neuron.reset()
    sample_output = model_neuron.sample(
        input_ids,
        sequence_length=seqlen,
        top_k=20
    )
    end = time.time()

    print('\nGenerated outputs:\n')
    for result in [tokenizer.decode(tok, skip_special_tokens=True) for tok in sample_output]:
        m = re.search(r"(.*)\n", result)
        if m:
            print(m.group(1))
        else:
            print("<unknown>")
    
    throughput = args.bs*new_tokens / (end-start)
    latency = (end - start)
    print(f'\nthroughput: {throughput:.1f} tokens/sec')
    print(f'batch latency: {latency:.2f} s')

while True:
    try:
        prompt = input(f"\nEnter prompt (or CTRL-D to exit): ")
        text = [prompt for _ in range(args.bs)]
        print('\nRunning inference ...')
        sample(model_neuron, text, args.seqlen)
    except EOFError:
        print()
        break
