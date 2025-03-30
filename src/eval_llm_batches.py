import json
import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
from peft import PeftModel
import os
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np

def np_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    
def parse_response(response_text):
    """
    Given a response text in the new format:
      "LOC - China
       MISC - newcomers
       ORG - Uzbekistan
       PER - none"
       
    This function returns a dictionary, for example:
      { "LOC": ["China"], "MISC": ["newcomers"], "ORG": ["Uzbekistan"] }
      
    Lines with an entity value of "none" are skipped.
    """
    entities = {}
    for line in response_text.splitlines():
        if " - " in line:
            entity_type, entity_value = line.split(" - ", 1)
            entity_type = entity_type.strip().upper()
            entity_value = entity_value.strip()
            if entity_value.lower() == "none":
                continue

            mentions = [e.strip() for e in entity_value.split(",") if e.strip()]
            if mentions:
                if entity_type in entities:
                    entities[entity_type].extend(mentions)
                else:
                    entities[entity_type] = mentions
    return entities

def get_bio_tags(sentence, entities):
    """
    Convert a sentence and its extracted entity mentions (dictionary) into token-level BIO tags.
    
    Parameters:
      sentence (str): The input sentence.
      entities (dict): Mapping of entity types to a list of mention strings.
    
    Returns:
      tokens (list): List of tokens (using whitespace split).
      tags (list): List of BIO tags in the same order as tokens.
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
                    break  # Mark only the first occurrence
    return tokens, tags

def evaluate_ner_pipeline_conll03(model, tokenizer, test_dataset, label_list, max_new_tokens=150, max_length=512, batch_size=8):
    """
    Evaluate NER performance on the CoNLL03 test set using batched text-generation.
    
    Parameters:
      model: The (fine-tuned or pre-trained) model.
      tokenizer: The corresponding tokenizer.
      test_dataset: The CoNLL03 test dataset loaded from Hugging Face.
      label_list: A list mapping numeric ner_tags to string labels.
      max_new_tokens (int): Maximum tokens to generate.
      max_length (int): Maximum prompt token length.
      batch_size (int): Number of examples to process in each batch.
    
    Returns:
      A dictionary with keys "precision", "recall", "f1", and "classification_report" (as a dict).
    """
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer
    )
    
    all_gold_tags = []
    all_pred_tags = []
    prompts_batch = []
    gold_sentences_batch = []
    generated_results = []
    
    for example in tqdm(test_dataset, desc="Evaluating CoNLL03"):
        sentence = " ".join(example["tokens"])
        gold_tags = [label_list[tag] for tag in example["ner_tags"]]
        #prompt = f"### Instruction:\nExtract named entities from the sentence: '{sentence}'\n\n### Response:"
        prompt = f"### Instruction:\nYou are an expert of natural language processing annotation, given a sentence, you are going to identify and classify each named entity according to its type: LOC (Location), MISC (Miscellaneous), ORG (Organization), or PER (Person).: '{sentence}'\n\n### Response:"

        prompts_batch.append(prompt)
        gold_sentences_batch.append((sentence, gold_tags))
        
        if len(prompts_batch) == batch_size:
            outputs = generator(
                prompts_batch,
                max_new_tokens=max_new_tokens,
                truncation=True,
                do_sample=False
            )
            for out, (sentence, gold_tags) in zip(outputs, gold_sentences_batch):
                generated_text = out[0].get("generated_text", "")
                if "### Response:" in generated_text:
                    pred_text = generated_text.split("### Response:")[-1].strip()
                else:
                    pred_text = generated_text.strip()
                generated_results.append({
                    "prompt": sentence,
                    "generated_response": pred_text
                })
                pred_entities = parse_response(pred_text)
                _, pred_tags = get_bio_tags(sentence, pred_entities)
                all_gold_tags.append(gold_tags)
                all_pred_tags.append(pred_tags)
            prompts_batch = []
            gold_sentences_batch = []
    
    if prompts_batch:
        outputs = generator(
            prompts_batch,
            max_new_tokens=max_new_tokens,
            truncation=True,
            do_sample=False
        )
        for out, (sentence, gold_tags) in zip(outputs, gold_sentences_batch):
            generated_text = out[0].get("generated_text", "")
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
    report = classification_report(all_gold_tags, all_pred_tags, output_dict=True)
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }
    return metrics, generated_results

def main():
    experiment_name = "QLoRA_Evaluation"
    run_name = "Evaluation_Run_finetuned_model"
    load_dotenv()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)
    else:
        raise RuntimeError("no hf token")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        mlflow.log_param("model_name", model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        base_model.eval()
        # load the fine-tuned adapter LoRA into the base model
        model = PeftModel.from_pretrained(base_model, "./output/qlora-ner-output-10/checkpoint-8770")
        model.eval()
        model = model.merge_and_unload()

        test_dataset = load_dataset("conll2003", split="test", trust_remote_code=True).select(range(1000))
        label_list = test_dataset.features["ner_tags"].feature.names

        metrics, generated_results = evaluate_ner_pipeline_conll03(model, tokenizer, test_dataset, label_list, batch_size=8)
        print("Evaluation Metrics:")

        metrics_filename = "output/eval/evaluation_metrics_peft.json"
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4, default=np_encoder)

        responses_filename = "output/eval/llm_generated_responses_peft.json"
        with open(responses_filename, "w") as f:
            json.dump(generated_results, f, indent=4, default=np_encoder)

        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])

        for entity, scores in metrics["classification_report"].items():
            if isinstance(scores, dict):
                for score_name, value in scores.items():
                    mlflow.log_metric(f"{entity}_{score_name}", value)
        
        mlflow.log_artifact(metrics_filename)
        mlflow.log_artifact(responses_filename)
        
        
if __name__ == "__main__":
    main()
