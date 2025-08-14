FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies for ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# First copy ONLY setup and requirements files
COPY setup.py .
COPY requirements.txt .

# Install dependencies (including editable install)
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the entire application
COPY . .

ARG PORT=7860
ENV PORT=$PORT

# Configure for headless server environment
ENV STREAMLIT_SERVER_HEADLESS=true

# Use PORT environment variable in command
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT"]