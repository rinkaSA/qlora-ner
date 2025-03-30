FROM --platform=linux/amd64 nvidia/cuda:11.7.1-base-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools && \
    pip install \
        torch \
        transformers \
        datasets \
        peft \
        trl \
        bitsandbytes \
        codecarbon \
        psutil \
        huggingface_hub \
        tensorboard \
        mlflow \
        seqeval \
        tqdm \
        python-dotenv \
        numpy

ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers

WORKDIR /workspace

COPY src/ ./src/
COPY datasets/ ./datasets/

CMD ["python", "src/quantize_train.py"]

