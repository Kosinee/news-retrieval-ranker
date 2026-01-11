FROM python:3.12-slim

WORKDIR /app

# Build tools needed for pytrec-eval-terrier wheel
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY config /app/config
COPY static/artifacts /app/static/artifacts

ENTRYPOINT ["python", "-m", "src.predict"]
