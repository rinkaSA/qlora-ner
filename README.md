# QLoRA-NER Project Overview

## Training Workflow

Based on paper: https://arxiv.org/pdf/2305.14314

- **Data & Preprocessing:**
  - Dataset: `datasets/ner_instruct_data.json`
  - Each example is preprocessed to combine an instruction and a structured response into a single text prompt using the true labels of train dataset partition.
  

### Model & Tokenizer Initialization

- **Model Loading:**  
  - The base model is **LLaMA-2 7B Chat** from Meta.
  - The model is loaded using a 4-bit quantization configuration (via `BitsAndBytesConfig`) which helps in reducing memory usage while maintaining performance.
  
- **Tokenizer:**  
  - The corresponding tokenizer is initialized to properly tokenize and pad input text.
  - The tokenizer’s `eos_token` is set as the padding token.

### Quantization and Efficient Fine-Tuning with LoRA

- **Quantization:**  
  - The model is loaded with 4-bit quantization, using NF4 quantization type and double quantization for improved precision.
  
- **Preparation for k-bit Training:**  
  - The model is preprocessed with `prepare_model_for_kbit_training` to enable quantization-aware training.

- **LoRA Adapters:**  
  - LoRA (Low-Rank Adaptation) is used for parameter-efficient fine-tuning.  
  - LoRA adapters introduce additional low-rank matrices to specific parts of the network (targeting `"q_proj"` and `"v_proj"` modules) rather than updating all model weights.
  - Key configuration parameters include:
    - **r:** Rank of the low-rank matrices.
    - **lora_alpha:** Scaling factor.
    - **lora_dropout:** Dropout rate to help regularize the adapters.
  - This approach reduces the training burden by focusing updates only on the added parameters while the pre-trained weights remain largely frozen.

## Container Creation & Deployment

- **Local Docker Build:**
  - A **Dockerfile** defines the environment:
    - **Base Image:** Uses an NVIDIA CUDA image (`nvidia/cuda:11.7.1-base-ubuntu20.04`)
    - **Dependencies:** Installs system packages and Python libraries (Torch, Transformers, etc.)
    - **Working Directory:** Set to `/workspace` and copies project folders (`src/` and `datasets/`)
    - **Default Command:** Runs the training script.
  - **Build Command:**
    ```bash
    docker build --platform linux/amd64 -t lora-train:latest . (or lora-train-eval -> the newer container versions with broader set of libraries)
    ```

- **Pushing & Pulling:**
  - **Push to Docker Hub:**
    - Tag the image (`irv12/lora-train:latest`) and push it. ( OR `irv12/lora-train-eval:latest`)
    ```bash
    docker tag lora-train:latest irv12/lora-train:latest
    docker push irv12/lora-train:latest
    ```
  - **On the HPC Cluster:**
    - Pull the Docker image and convert it into a Singularity image:
    ```bash
    singularity build lora-train.sif docker://irv12/lora-train:latest
    ```

- **Running on the Cluster:**
  - **SLURM Job Script:** A SLURM script runs the Singularity container with GPU support (using `--nv`) and bind-mounts host project directory (`/data/horse/ws/irve354e-uniNer_test/qlora-ner/qlora-ner`) to `/workspace` inside the container.
  - **Container Lifecycle:** The container starts, executes the training and stops after the job ends. Output data written to `/workspace` is persisted on the host via the bind mount.

# Evaluation of the Fine-Tuned NER Model

This section of the script evaluates the performance of the both fine-tuned LLaMA-2 7B Chat model on a Named Entity Recognition (NER) task using the CoNLL03 test dataset and base model. The evaluation workflow combines text generation, response parsing, BIO tagging, and metric computation, all integrated into an MLflow experiment for tracking.

---

## Key Concepts

- **Text Generation Pipeline:**  
  The evaluation leverages a text-generation pipeline to prompt the model with sentences from the CoNLL03 dataset. The prompt instructs the model to extract and classify named entities ( LOC, ORG, PER, MISC) in each sentence.

- **Response Parsing and BIO Tagging:**  
  After generation, the raw output is parsed to extract entity mentions. These mentions are then converted into token-level BIO tags, which are compared against the true labels from the dataset.

- **Metric Computation:**  
  Using standard NER evaluation metrics (precision, recall, F1-score, and a detailed classification report), the script quantifies the performance of the model on the test set.

- **Batch Processing:**  
  To efficiently handle the dataset, the evaluation processes examples in batches. Each batch is sent through the generation pipeline, and the responses are parsed and tagged accordingly. (though the batch size should be increased..)

- **MLflow Integration:**  
  The entire evaluation process is tracked with MLflow. Key parameters, computed metrics, and generated responses are logged and saved as artifacts.

---

## Workflow Summary

1. **Model Preparation:**  
   The fine-tuned model (with merged LoRA adapters) is loaded in a quantized state for evaluation.

2. **Data Loading:**  
   A subset of the CoNLL03 test dataset is loaded and pre-processed, extracting the sentence tokens and corresponding gold NER tags.

3. **Prompting and Generation:**  
   For each test sentence, a prompt is created to instruct the model to perform NER. The text-generation pipeline outputs the generated response for each prompt.

4. **Parsing and Tagging:**  
   The generated responses are parsed to extract named entities, which are then converted into BIO tags matching the tokenization of the input sentences.

5. **Evaluation Metrics:**  
   The predicted BIO tags are compared with the ground-truth tags to compute precision, recall, F1-score, and a detailed classification report using the `seqeval` library. Results are compared fro inference of the model before quantized training (base model) and after. 

6. **Logging Results:**  
   Evaluation metrics and generated responses are saved as JSON files and logged into MLflow, along with other run parameters, to ensure a comprehensive record of the evaluation.

