from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
import os
from datasets import load_dataset

from dotenv import load_dotenv
from huggingface_hub import login

def preprocess_example(example):
    instruction = example.get("instruction", "")
    response = example.get("response", {})
    response_parts = []
    for entity_type, entities in response.items():
        if entities is None:
            continue 
        if isinstance(entities, list):
            entities_str = ", ".join(entities)
        else:
            entities_str = str(entities)
        response_parts.append(f"{entity_type.upper()}: {entities_str}")
    response_str = "; ".join(response_parts)
    full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response_str}"
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
    output_dir="./qlora-ner-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=True,
    logging_dir="./logs",
    save_strategy="epoch",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

trainer.train()
