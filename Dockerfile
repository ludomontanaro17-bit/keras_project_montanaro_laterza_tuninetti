# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Assicura che la cartella model esista (per sicurezza)
RUN mkdir -p model

EXPOSE 5000

CMD ["python", "api.py"]