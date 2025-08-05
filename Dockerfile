FROM python:3.12-slim

# Reteti dependencies:
RUN pip install --no-cache \
    duckdb                 \
    minio                  \
    pyarrow                \
    tokenizers

# Demo-related dependencies:
RUN pip install --no-cache \
    datasets               \
    "gradio <= 5.34.0"     \
    pandas                 \
    python-dotenv

RUN apt-get update && apt-get install -y curl

RUN mkdir /home/reteti

RUN curl -o /home/reteti/stopwords-iso.json \
    https://raw.githubusercontent.com/stopwords-iso/stopwords-iso/master/stopwords-iso.json

# Reteti files:
COPY ./reteti_core.py   /home/reteti/reteti_core.py
COPY ./reteti_text.py   /home/reteti/reteti_text.py

# Demo application files:
COPY ./.env             /home/reteti/.env
COPY ./demo_searcher.py /home/reteti/demo_searcher.py

# Start the demo application by default:
EXPOSE 7860
CMD ["python", "/home/reteti/demo_searcher.py"]

# docker build -t reteti-demo .
# docker buildx build -t reteti-demo .
