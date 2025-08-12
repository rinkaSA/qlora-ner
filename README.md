# QLoRA-NER Project: Enhancing Named Entity Recognition with Quantized LoRA

## Executive Summary

This project demonstrates how Quantized Low-Rank Adaptation (QLoRA) training can dramatically improve performance on Named Entity Recognition (NER) tasks when applied to Large Language Models. Using LLaMA-2 7B chat model, we achieved:

- **83.4% F1 score** with QLoRA fine-tuning + few-shot prompting
- **59.0% F1 score** with QLoRA fine-tuning + basic prompting
- **24.8% F1 score** with few-shot prompting only (no fine-tuning)
- **10.9% F1 score** baseline (no fine-tuning, basic prompting)

These results clearly demonstrate that combining QLoRA fine-tuning with effective prompt engineering yields the best performance on NER tasks, significantly outperforming either technique alone.

## Project Overview

This project is based on the QLoRA approach introduced in the paper: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)

Our goal was to demonstrate that QLoRA training (quantized LoRA adapters training) for LLaMA-2 7B chat model can significantly increase the F1 score on Named Entity Recognition tasks.

## Study Design

We implemented QLoRA training using:
- **Dataset**: CoNLL-03 (train split for fine-tuning, test split for evaluation)
- **Base Model**: LLaMA-2 7B chat
- **Task**: Named Entity Recognition (LOC, MISC, ORG, PER entity types)

### Prompt Engineering

We experimented with two different prompting strategies:

1. **Basic Prompt**:
   ```
   You are an expert in natural language processing annotation. Given a sentence, identify and classify each named entity into one of the following types: LOC (Location), MISC (Miscellaneous), ORG (Organization), or PER (Person).
   ```

2. **Few-Shot Prompt**:
   ```
   You are an expert in natural language processing annotation. Given a sentence, identify and classify each named entity into one of the following types: LOC (Location), MISC (Miscellaneous), ORG (Organization), or PER (Person).

   For example, consider the sentence:
   'Brazilian Planning Minister Antonio Kandir will submit to a draft copy of the 1997 federal budget to Congress on Thursday, a ministry spokeswoman said.'

   Expected output: {'MISC': ['Brazilian'], 'PER': ['Antonio Kandir'], 'ORG': ['Congress']}

   Now, given the sentence: {}
   ```

## Experimental Results

### 1. Base Model (No Fine-tuning, Basic Prompt)

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Overall  | 0.122     | 0.098  | 0.109    | -       |
| LOC      | 0.581     | 0.138  | 0.224    | 130     |
| MISC     | 0.061     | 0.057  | 0.059    | 35      |
| ORG      | 0.158     | 0.061  | 0.088    | 49      |
| PER      | 0.850     | 0.088  | 0.160    | 193     |

### 2. Fine-tuning with Basic Prompt

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Overall  | 0.646     | 0.543  | 0.590    | -       |
| LOC      | 0.816     | 0.477  | 0.602    | 130.0   |
| MISC     | 0.238     | 0.143  | 0.179    | 35.0    |
| ORG      | 0.579     | 0.449  | 0.506    | 49.0    |
| PER      | 0.706     | 0.684  | 0.695    | 193.0   |

### 3. Fine-tuning with Few-Shot Prompt

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Overall  | 0.933     | 0.754  | 0.834    | -       |
| LOC      | 0.944     | 0.908  | 0.925    | 130.0   |
| MISC     | 0.704     | 0.543  | 0.613    | 35.0    |
| ORG      | 0.912     | 0.633  | 0.747    | 49.0    |
| PER      | 0.986     | 0.720  | 0.832    | 193.0   |

### 4. Base Model with Few-Shot Prompt (No Fine-tuning)

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Overall  | 0.232     | 0.265  | 0.248    | -       |
| LOC      | 0.388     | 0.254  | 0.307    | 130.0   |
| MISC     | 0.061     | 0.400  | 0.106    | 35.0    |
| ORG      | 0.135     | 0.102  | 0.116    | 49.0    |
| PER      | 0.487     | 0.290  | 0.364    | 193.0   |

### Key Findings

1. **QLoRA Fine-tuning Significantly Improves Performance**: Fine-tuning with QLoRA improved F1 score from 10.9% to 59.0% even with basic prompting.

2. **Prompt Engineering Enhances Results**: Few-shot prompting alone improved the base model F1 score from 10.9% to 24.8%.

3. **Combined Approach is Best**: The combination of QLoRA fine-tuning with few-shot prompting yielded the highest F1 score of 83.4%.

4. **Entity Type Performance**: The model performs best on LOC and PER entities, while MISC entities remain the most challenging.

## Technical Implementation

### Data & Preprocessing

- **Dataset**: `datasets/ner_instruct_data.json`
- Each example combines an instruction and a structured response into a single text prompt using the true labels from the train dataset partition.

### Model & Tokenizer Initialization

- **Model Loading**:  
  - Base model: **LLaMA-2 7B Chat** from Meta
  - Loaded using 4-bit quantization configuration (via `BitsAndBytesConfig`) to reduce memory usage while maintaining performance
  
- **Tokenizer**:  
  - Initialized to properly tokenize and pad input text
  - The tokenizer's `eos_token` is set as the padding token

### Quantization and Efficient Fine-Tuning with LoRA

- **Quantization**:  
  - 4-bit quantization with NF4 type and double quantization for improved precision
  
- **Preparation for k-bit Training**:  
  - Model preprocessed with `prepare_model_for_kbit_training` to enable quantization-aware training

- **LoRA Adapters**:  
  - LoRA (Low-Rank Adaptation) used for parameter-efficient fine-tuning
  - Introduces additional low-rank matrices to specific network parts (targeting `q_proj` and `v_proj` modules)
  - Key configuration parameters:
    - **r**: Rank of the low-rank matrices
    - **lora_alpha**: Scaling factor
    - **lora_dropout**: Dropout rate for regularization
  - Reduces training burden by focusing updates only on added parameters while pre-trained weights remain frozen

## Deployment Pipeline

### Container Creation

- **Docker Configuration**:
  - **Base Image**: NVIDIA CUDA image (`nvidia/cuda:11.7.1-base-ubuntu20.04`)
  - **Dependencies**: System packages and Python libraries (PyTorch, Transformers, etc.)
  - **Working Directory**: `/workspace` with project folders (`src/` and `datasets/`)
  - **Default Command**: Training script

- **Build Command**:
  ```bash
  docker build --platform linux/amd64 -t lora-train:latest .
  # Or for newer versions with broader library support:
  # docker build --platform linux/amd64 -t lora-train-eval:latest .
  ```

### Deployment Workflow

- **Push to Docker Hub**:
  ```bash
  docker tag lora-train:latest irv12/lora-train:latest
  docker push irv12/lora-train:latest
  ```

- **HPC Cluster Deployment**:
  ```bash
  # Convert Docker image to Singularity
  singularity build lora-train.sif docker://irv12/lora-train:latest
  ```

- **Execution**:
  - SLURM job script runs the Singularity container with GPU support (`--nv`)
  - Bind-mounts host project directory to `/workspace` inside the container
  - Container executes training and persists output data via the bind mount

## Evaluation Framework

The evaluation process assesses both the base and fine-tuned LLaMA-2 7B Chat models on NER tasks using the CoNLL03 test dataset.

### Evaluation Pipeline

1. **Text Generation**:
   - Prompts the model with sentences from CoNLL03
   - Instructs the model to extract and classify named entities

2. **Response Processing**:
   - Parses raw outputs to extract entity mentions
   - Converts mentions into token-level BIO tags
   - Compares against ground truth labels

3. **Metrics Calculation**:
   - Precision, recall, F1-score per entity type
   - Overall performance metrics
   - Detailed classification reports

4. **Optimization**:
   - Batch processing for efficiency
   - MLflow integration for experiment tracking

### Evaluation Workflow

1. **Model Preparation**: Load fine-tuned model with merged LoRA adapters in quantized state

2. **Data Processing**: Load and preprocess CoNLL03 test dataset

3. **Inference**: Generate NER predictions for test sentences

4. **Analysis**: Convert predictions to BIO format and compute metrics

5. **Reporting**: Log results and generated responses to MLflow
