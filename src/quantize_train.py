from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
import os
from datasets import load_dataset

from dotenv import load_dotenv
from huggingface_hub import login

from codecarbon import EmissionsTracker
import time

def preprocess_example(example):
    full_text = example["text"]
    tokenized = tokenizer(full_text, truncation=True, max_length=512)
    tokenized["text"] = full_text
    return tokenized


load_dotenv()
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    raise RuntimeError("no hf token")
model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="/data/horse/ws/irve354e-uniNer_test/qlora-ner/qlora-ner/datasets/ner_instruct_data.json", split="train")
dataset = dataset.map(preprocess_example)

training_args = TrainingArguments(
    output_dir="./output/qlora-ner-output-4",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=10,
    fp16=True,
    logging_dir="./output/logs",
    save_strategy="epoch",
    report_to="none" # change to tensorboard but docker rebuilt is needed to include library
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

tracker = EmissionsTracker(project_name="qlora-ner-finetuning")
tracker.start()
start_time = time.time()
trainer.train()
end_time = time.time()
tracker.stop()
print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")
