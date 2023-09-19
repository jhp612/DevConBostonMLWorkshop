import torch, torch_xla
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, TrainingArguments
from optimum.neuron import TrainiumTrainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import load_from_disk
import os

ds = load_from_disk("poems_dataset_64")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
print(f"# model params: {sum([p.numel() for p in model.parameters()]):,}")
col = DataCollatorForLanguageModeling(tokenizer, mlm=False)

print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 1)}")
if os.environ.get("NEURON_PARALLEL_COMPILE", None):
    max_steps = 20
else:
    max_steps = 1000

training_args = TrainingArguments(
    output_dir="./output/",
    overwrite_output_dir=True,
    num_train_epochs=10,
    max_steps=max_steps,
    warmup_steps=300,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=50,
    logging_steps=10,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    report_to="tensorboard",
    dataloader_drop_last=True
)

trainer = TrainiumTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=col,
)

trainer.train()
