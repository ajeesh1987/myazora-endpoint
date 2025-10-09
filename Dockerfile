# ✅ CUDA 12.1 base with PyTorch 2.3.1
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Basic system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ✅ Force clean pip + torch reinstall
RUN pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchaudio || true
RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ✅ Now install compatible AI + web libs
RUN pip install --no-cache-dir \
    diffusers==0.29.0 \
    transformers==4.42.3 \
    accelerate==0.31.0 \
    Pillow==10.3.0 \
    fastapi==0.110.0 \
    uvicorn==0.29.0 \
    requests

# Copy your code
COPY . /app
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8000"]
