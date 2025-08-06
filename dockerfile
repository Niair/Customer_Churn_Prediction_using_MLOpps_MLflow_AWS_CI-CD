FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE $PORT

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=$PORT"]
