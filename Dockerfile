# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final minimal execution image
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy python packages installed globally from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

# Ensure local packages are on PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application layers, feature definitions, and generated mlrun stores
COPY src/ ./src
COPY feature_store/ ./feature_store
COPY mlruns/ ./mlruns

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]