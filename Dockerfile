# Use an official PyTorch image with CUDA for GPU acceleration
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install basic utilities
RUN apt-get update && apt-get install -y git curl ffmpeg libgl1-mesa-glx

# Copy dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY . .

# Expose port (used by Uvicorn or FastAPI)
EXPOSE 3000

# Launch FastAPI using Uvicorn
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "3000"]
