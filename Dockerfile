FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Torch + CUDA support early for caching
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Add code
COPY . /app
WORKDIR /app

EXPOSE 3000
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "3000"]
