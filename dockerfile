# Lightweight Python base
FROM python:3.11-slim

WORKDIR /app

# Install only required system deps (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements
COPY requirements.txt .

# Install only inference dependencies (scikit-learn, pandas, streamlit, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what’s needed for inference
COPY app.py .
COPY src/pipeline/ ./src/pipeline/
COPY src/utils.py ./src/
COPY src/exception.py ./src/
COPY src/logger.py ./src/
COPY artifacts/model.pkl ./artifacts/model.pkl
COPY artifacts/preprocessor.pkl ./artifacts/preprocessor.pkl
# COPY artifacts/ ./artifacts/  # for both training + prediction

# Expose Streamlit port
EXPOSE 7860

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]


















# (For both training + prediction)

## Use a lightweight Python image
#FROM python:3.11-slim
#
## Set working directory
#WORKDIR /app
#
## Install only necessary system dependencies
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential gcc curl \
#    && rm -rf /var/lib/apt/lists/*
#
## Copy requirements first (for caching)
#COPY requirements.txt .
#
## Install Python dependencies
#RUN pip install --no-cache-dir --upgrade pip \
#    && pip install --no-cache-dir -r requirements.txt
#
## Copy the application code
#COPY . .
#
## Copy trained model artifacts
#COPY artifacts/ /app/artifacts/
#
## Expose port for Hugging Face Spaces (uses $PORT automatically)
#EXPOSE 7860
#
## Command to run Streamlit app
#CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
#