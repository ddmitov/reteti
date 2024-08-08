FROM python:3.12-slim

# Reteti core modules:
RUN pip install --no-cache \
    deltalake              \
    duckdb                 \
    huggingface-hub        \
    minio                  \
    pandas                 \
    pyarrow                \
    pysbd                  \
    python-dotenv          \
    tokenizers

# Tokenizer:
RUN mkdir     /tokenizer
RUN chmod 777 /tokenizer

COPY ./tokenizer_downloader.py /etc/tokenizer_downloader.py
RUN python3 /etc/tokenizer_downloader.py

# Reteti Gradio demo application:
RUN pip install --no-cache gradio

RUN mkdir /home/reteti
COPY ./.env        /home/reteti/.env
COPY ./reteti.py   /home/reteti/reteti.py
COPY ./searcher.py /home/reteti/searcher.py

# Start Reteti Gradio demo application by default:
EXPOSE 7860
CMD ["python", "/home/reteti/searcher.py"]

# docker build -t reteti .
# docker buildx build -t reteti .
