FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies for ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy only the files needed for installation first.
# This allows Docker to cache the installed packages layer.
COPY requirements.txt setup.py ./

# 2. Install the project.
# The command 'pip install .' will execute setup.py.
# setup.py will then read requirements.txt, remove '-e .', and install the dependencies.
RUN pip install --no-cache-dir .

# 3. Now copy the rest of your application code.
# If you only change your app code (e.g., app.py), Docker will use the cache
# for the layers above and your build will be much faster.
COPY . .

# 4. Expose the port and run the application (this part remains the same).
ENV PORT=${PORT:-7860}
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT"]