# Production Docker image — pulls RoBERTa model from HF Hub and evaluates
#
# Build:
#   docker build --build-arg HF_REPO=Shikhar16/goodreads-genre-classifier -t genre-eval .
#
# Run:
#   docker run --rm genre-eval

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy evaluation script (downloads its own data — no pickle needed)
COPY evaluate_from_hub.py .

ARG HF_REPO="Shikhar16/goodreads-genre-classifier"
ENV HF_REPO=${HF_REPO}

RUN mkdir -p results

CMD ["sh", "-c", "python evaluate_from_hub.py --repo ${HF_REPO}"]