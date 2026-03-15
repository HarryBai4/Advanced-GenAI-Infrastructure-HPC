# Use NVIDIA PyTorch base for high-performance deep learning
FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL maintainer="Harry Bai"
LABEL description="GenAI Infrastructure HPC Dev Environment"

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install specific TensorRT-LLM requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["/bin/bash"]
