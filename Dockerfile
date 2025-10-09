# CUDA-ready PyTorch base
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# System libs for Pillow / Diffusers
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install compatible versions
COPY requirements.txt .
RUN pip install --no-cache-dir \
    diffusers==0.29.0 \
    transformers==4.42.3 \
    accelerate==0.31.0 \
    Pillow==10.3.0 \
    requests \
    fastapi \
    uvicorn

# Copy app
COPY . /app
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8000"]
