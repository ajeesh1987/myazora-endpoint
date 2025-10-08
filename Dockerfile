# Use the latest verified RunPod PyTorch image with CUDA and Torch preinstalled
FROM runpod/pytorch:2.1.0-py310-cu121

# Install system dependencies for Pillow and Diffusers
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy and install requirements (torch already included)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app
WORKDIR /app

EXPOSE 3000
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "3000"]
