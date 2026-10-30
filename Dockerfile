# ========================================================
# codeAtlas: Containerized Deployment Environment
# ========================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV TOKENIZERS_PARALLELISM=false

# Install essential system dependencies and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY mini/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

EXPOSE 8501

WORKDIR /app/mini

# Healthcheck to monitor Streamlit status
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch codeAtlas UI
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]

