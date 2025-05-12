FROM python:3.10

# Create and set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y git

# Copy dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all source code
COPY . .

# Expose the FastAPI port
EXPOSE 3000

# Run the FastAPI server
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "3000"]
