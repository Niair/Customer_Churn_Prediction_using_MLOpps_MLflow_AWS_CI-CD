FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies for ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=${PORT:-7860}

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT"]
