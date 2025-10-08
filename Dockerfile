# CUDA-ready PyTorch base from NVIDIA + PyTorch team (kept very lean)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Install small set of system libs needed by Pillow / Diffusers
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install (Torch already included in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app
WORKDIR /app

EXPOSE 3000
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "3000"]
