# ---- Stage 1: Build ----
# Use the full Python image to build dependencies, which has necessary build tools.
FROM python:3.11-slim-bullseye AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy only necessary files and install dependencies into the venv
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Final Image ----
# Use a slim image for the final product to keep it small.
FROM python:3.11-slim-bullseye

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code
COPY . .

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set Streamlit-specific environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Hugging Face Spaces provides the PORT environment variable.
# Default to 7860 if not provided.
EXPOSE 7860
ENV PORT=7860

# The command to run the app. This is the correct way to start the server.
# It uses the $PORT variable provided by Hugging Face Spaces.
CMD ["streamlit", "run", "app.py", "--server.port", "$PORT", "--server.address", "0.0.0.0"]