# Use official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY train.py .
COPY evaluate.py .

# Create directories for data and model output
RUN mkdir -p data/train data/test

# Default command (can be overridden at runtime)
CMD ["python", "evaluate.py"]