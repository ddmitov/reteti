FROM python:3.12-slim

# Reteti core modules:
RUN pip install --no-cache \
    duckdb                 \
    minio                  \
    pandas                 \
    pyarrow                \
    tokenizers

# Tokenizer:
RUN mkdir     /tokenizer
RUN chmod 777 /tokenizer

COPY ./tokenizer_downloader.py /etc/tokenizer_downloader.py
RUN  python3 /etc/tokenizer_downloader.py

# Demo-related modules:
RUN pip install --no-cache \
    gradio                 \
    huggingface-hub        \
    python-dotenv

# Gradio demo application:
RUN pip install --no-cache gradio

# RUN mkdir /home/reteti
COPY ./.env             /home/reteti/.env
COPY ./reteti_core.py   /home/reteti/reteti_core.py
COPY ./reteti_text.py   /home/reteti/reteti_text.py
COPY ./demo_searcher.py /home/reteti/demo_searcher.py

# Start the Gradio demo application by default:
EXPOSE 7860
CMD ["python", "/home/reteti/demo_searcher.py"]
