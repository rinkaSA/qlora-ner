import json
from datasets import load_dataset

DATASET_NAME = "conll2003"  
SPLIT = "train"
OUTPUT_FILE = "/data/horse/ws/irve354e-uniNer_test/qlora-ner/qlora-ner/datasets/ner_instruct_data.json"

dataset = load_dataset(DATASET_NAME, split=SPLIT)
label_names = dataset.features["ner_tags"].feature.names

def ner_to_instruction(example):
    words = example["tokens"]
    tags = example["ner_tags"]
    sentence = " ".join(words)
    
    entities = {}
    current_entity = []
    current_type = None

    for word, tag_id in zip(words, tags):
        tag = label_names[tag_id]
        if tag == "O":
            if current_entity:
                entity_str = " ".join(current_entity)
                entities.setdefault(current_type, []).append(entity_str)
                current_entity = []
                current_type = None
            continue

        prefix, ent_type = tag.split("-")
        if prefix == "B":
            if current_entity:
                entity_str = " ".join(current_entity)
                entities.setdefault(current_type, []).append(entity_str)
            current_entity = [word]
            current_type = ent_type
        elif prefix == "I" and current_type == ent_type:
            current_entity.append(word)
        else:
            if current_entity:
                entity_str = " ".join(current_entity)
                entities.setdefault(current_type, []).append(entity_str)
            current_entity = [word]
            current_type = ent_type

    if current_entity:
        entity_str = " ".join(current_entity)
        entities.setdefault(current_type, []).append(entity_str)
    
    expected_entities = ["LOC", "MISC", "ORG", "PER"]
    response = {etype: entities.get(etype, None) for etype in expected_entities}
    
    instruction = f"Extract named entities from the sentence: '{sentence}'"
    response_parts = []
    for entity_type in expected_entities:
        ent_list = response.get(entity_type)
        if ent_list is None:
            continue
        if isinstance(ent_list, list) and ent_list:
            entities_str = ", ".join(ent_list)
            response_parts.append(f"{entity_type.upper()}: {entities_str}")
    response_str = "; ".join(response_parts)
    
    full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response_str}"
    return {"text": full_text}

formatted_dataset = dataset.map(ner_to_instruction)

formatted_dataset = formatted_dataset.remove_columns(
    [col for col in formatted_dataset.column_names if col not in ["text"]]
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(formatted_dataset.to_list(), f, indent=2, ensure_ascii=False)
