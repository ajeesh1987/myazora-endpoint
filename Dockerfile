# Use RunPod's prebuilt CUDA + PyTorch base (already includes CUDA + Torch)
FROM runpod/pytorch:3.10-2.1.0

# Install system deps needed for Pillow / Diffusers
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (no torch here — already in base)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app
WORKDIR /app

EXPOSE 3000
CMD ["uvicorn", "handler:app", "--ho]()
