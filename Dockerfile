FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && \
    pip install --upgrade pip

# Install heavyweight deps first to cache
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install the rest
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 3000
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "3000"]
