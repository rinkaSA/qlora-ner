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

    return {
        "instruction": f"Extract named entities from the sentence: '{sentence}'",
        "response": entities  
    }

formatted_dataset = dataset.map(ner_to_instruction)

final_dataset = formatted_dataset.remove_columns(
    [col for col in formatted_dataset.column_names if col not in ["instruction", "response"]]
)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_dataset.to_list(), f, indent=2, ensure_ascii=False)

