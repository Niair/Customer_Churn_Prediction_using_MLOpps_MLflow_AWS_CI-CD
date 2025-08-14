FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies for ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# First copy ONLY setup and requirements files
COPY setup.py ./
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the entire application
COPY . .

# Streamlit runs on a dynamic port set by HF
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose $PORT (Hugging Face sets it automatically)
EXPOSE $PORT

# Always bind to 0.0.0.0 and use HF's dynamic port
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0