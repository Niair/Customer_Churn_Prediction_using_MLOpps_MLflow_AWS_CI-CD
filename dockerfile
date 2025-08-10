FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies for ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

COPY . .
RUN pip install .

EXPOSE 5000
ENV PORT=5000

CMD ["streamlit", "run", "./app.py", "--server.address=0.0.0.0", "--server.port=5000"]
