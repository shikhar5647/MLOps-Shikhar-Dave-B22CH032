# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir \
    transformers \
    torch \
    sacrebleu \
    sentencepiece

# Run evaluation script
CMD ["python", "translate.py"]