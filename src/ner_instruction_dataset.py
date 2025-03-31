import json
from datasets import load_dataset

DATASET_NAME = "conll2003"  
SPLIT = "train"
OUTPUT_FILE = "/data/horse/ws/irve354e-uniNer_test/qlora-ner/qlora-ner/datasets/ner_instruct_data_fewshot_cot.json"

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
    response = {etype: entities.get(etype, []) for etype in expected_entities} # empty lists are filtered out
    formatted_response = {k: v for k, v in response.items() if v}
    instruction = (
        "You are an expert in natural language processing annotation. Given a sentence, " # 1 (w/o stating enitity types) 2 with
        "identify and classify each named entity into one of the following types: "
        "LOC (Location), MISC (Miscellaneous), ORG (Organization), or PER (Person).\n\n"
        "For example, consider the sentence:\n" ### FEW SHOT START  3
        "'Brazilian Planning Minister Antonio Kandir will submit to a draft copy of the 1997 federal budget to Congress on Thursday, a ministry spokeswoman said.'\n\n"
        "Expected output: {'MISC': ['Brazilian'], 'PER': ['Antonio Kandir'], 'ORG': ['Congress']}\n\n" ### FEW SHOT END 3
        f"Now, given the sentence: {sentence}" # 1 2
    )


    """
    FOR CoT and few shot. but then we need to create a rsponse with reasoning? not sure how to proceed here
    instruction = (
        "You are an expert in natural language processing annotation. Given a sentence, " # 1 (w/o stating enitity types) 2 with
        "identify and classify each named entity into one of the following types: "
        "LOC (Location), MISC (Miscellaneous), ORG (Organization), or PER (Person).\n\n"
        "Follow these steps to annotate the sentence. \n" ### COT START - 4
        "Step 1.#### Read the sentence and understand its context.\n"
        "Step 2.#### Identify potential named entities within the sentence.\n"
        "Step 3.#### Determine the type of each entity (LOC, MISC, ORG, PER) based on the context.\n"
        "Step 4.#### Justify the classification of each entity with reasoning. \n\n"
        "Use the following format:\n"
        "Step 1.#### <step 1 reasoning>\n"
        "Step 2.#### <step 2 reasoning>\n"
        "Step 3.#### <step 3 reasoning>\n"
        "Step 4.#### <final output> \n"
        "Make sure to include #### to separate every step.\n" ### COT END -4
        "For example, consider the sentence:\n" ### FEW SHOT START  3
        "Sentence: 'Brazilian Planning Minister Antonio Kandir will submit to a draft copy of the 1997 federal budget to Congress on Thursday , a ministry spokeswoman said .\n"
        "Step 1.#### The sentence describes an action by Antonio Kandir, the Brazilian Planning Minister, who is planning to submit a draft of the 1997 federal budget to Congress, as stated by a ministry spokeswoman.\n"
        "Step 2.#### The entities identified are Brazilian (as an adjective related to Antonio Kandir), Antonio Kandir, and Congress.\n"
        "Step 3.#### The term 'Brazilian' is associated with Antonio Kandir and is classified as miscellaneous (MISC), as it describes a nationality. Antonio Kandir is classified as a person (PER), as it is an individual's name. Congress is classified as an organization (ORG), as it refers to a governmental legislative body.\n"
        "Step 4.#### {{'MISC': ['Brazilian'], 'PER': ['Antonio Kandir'], 'ORG': ['Congress']}}"
        f"Now, given the sentence: {sentence}" )
    """
    response_str = str(formatted_response)
    
    full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response_str}"
    return {"text": full_text}

formatted_dataset = dataset.map(ner_to_instruction)

formatted_dataset = formatted_dataset.remove_columns(
    [col for col in formatted_dataset.column_names if col not in ["text"]]
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(formatted_dataset.to_list(), f, indent=2, ensure_ascii=False)
