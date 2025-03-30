import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
from dotenv import load_dotenv
import json
load_dotenv()

def parse_response(response_text):
    """
    Given a response text in the format:
      "PER: Alice, Bob; ORG: Acme Inc, Globex"
    Returns:
      A dictionary: { "PER": ["Alice", "Bob"], "ORG": ["Acme Inc", "Globex"] }
    """
    entities = {}
    response_text = response_text.strip()
    for part in response_text.split(";"):
        if ":" in part:
            entity_type, entity_list = part.split(":", 1)
            entity_type = entity_type.strip().upper()
            mentions = [e.strip() for e in entity_list.split(",") if e.strip()]
            if mentions:
                entities[entity_type] = mentions
    return entities

def get_bio_tags(sentence, entities):
    """
    Convert a sentence and its extracted entity mentions (dictionary) into token-level BIO tags.
    
    Parameters:
      sentence (str): The input sentence.
      entities (dict): Mapping of entity types to a list of mention strings.
    
    Returns:
      tokens (list): List of tokens (obtained by whitespace splitting).
      tags (list): List of BIO tags in the same order as tokens.
      
    Note: This function uses a simple first-match approach on a whitespace-tokenized sentence.
    """
    tokens = sentence.split()
    tags = ["O"] * len(tokens)
    
    for ent_type, mentions in entities.items():
        for mention in mentions:
            mention_tokens = mention.split()
            for i in range(len(tokens) - len(mention_tokens) + 1):
                if tokens[i:i+len(mention_tokens)] == mention_tokens:
                    tags[i] = "B-" + ent_type
                    for j in range(1, len(mention_tokens)):
                        tags[i+j] = "I-" + ent_type
                    break  # mark the first occurrence
    return tokens, tags

def evaluate_ner_pipeline_conll03(model, tokenizer, test_dataset, label_list, max_new_tokens=150, max_length=512):
    gen_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer)
    
    all_gold_tags = []
    all_pred_tags = []
    
    for example in tqdm(test_dataset, desc="Evaluating CoNLL03"):
        sentence = " ".join(example["tokens"])
        gold_tags = [label_list[tag] for tag in example["ner_tags"]]
        prompt = f"### Instruction:\nExtract named entities from the sentence: '{sentence}'\n\n### Response:"
        
        result = gen_pipeline(
            prompt, 
            max_new_tokens=max_new_tokens, 
            truncation=True, 
            do_sample=False
        )[0]
        generated_text = result.get("generated_text", "")
        
        if "### Response:" in generated_text:
            pred_text = generated_text.split("### Response:")[-1].strip()
        else:
            pred_text = generated_text.strip()
        
        pred_entities = parse_response(pred_text)
        _, pred_tags = get_bio_tags(sentence, pred_entities)
        all_gold_tags.append(gold_tags)
        all_pred_tags.append(pred_tags)
    
    precision = precision_score(all_gold_tags, all_pred_tags)
    recall = recall_score(all_gold_tags, all_pred_tags)
    f1 = f1_score(all_gold_tags, all_pred_tags)
    report = classification_report(all_gold_tags, all_pred_tags)
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }
    
    return metrics

if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-chat-hf" 
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      device_map="auto",
      trust_remote_code=True)
  
    
    test_dataset = load_dataset("conll2003", split="test")
    label_list = test_dataset.features["ner_tags"].feature.names
    
    metrics = evaluate_ner_pipeline_conll03(model, tokenizer, test_dataset, label_list)
    print("Evaluation Metrics:")
    with open("output/eval/evaluation_metrics.json", "w") as f:
      json.dump(metrics, f, indent=4)