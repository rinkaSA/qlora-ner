# QLoRA-NER Project Overview

## Training Workflow

- **Data & Preprocessing:**
  - Dataset: `datasets/ner_instruct_data.json`
  - Each example is preprocessed to combine an instruction and a structured response into a single text prompt.
  

- **Model & Fine-Tuning:**
  - Base Model: LLaMA-2 7B Chat (quantized using QLoRA techniques)
  - Fine-Tuning: Uses PEFT with LoRA adapters to efficiently adapt the pre-trained model.
  - Trainer: Utilizes TRL’s `SFTTrainer` to perform supervised fine-tuning on the prepared dataset.
  - Training Script: Located in `src/quantize_train.py`, it handles data loading, tokenization, training, and logging metrics.

## Container Creation & Deployment

- **Local Docker Build:**
  - A **Dockerfile** defines the environment:
    - **Base Image:** Uses an NVIDIA CUDA image (`nvidia/cuda:11.7.1-base-ubuntu20.04`)
    - **Dependencies:** Installs system packages and Python libraries (Torch, Transformers, etc.)
    - **Working Directory:** Set to `/workspace` and copies project folders (`src/` and `datasets/`)
    - **Default Command:** Runs the training script.
  - **Build Command:**
    ```bash
    docker build --platform linux/amd64 -t lora-train:latest .
    ```

- **Pushing & Pulling:**
  - **Push to Docker Hub:**
    - Tag the image (`irv12/lora-train:latest`) and push it.
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
