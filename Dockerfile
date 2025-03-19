FROM python:3.12-slim

# Reteti dependencies:
RUN pip install --no-cache \
    duckdb                 \
    minio                  \
    pyarrow                \
    tokenizers

# Demo-related modules:
RUN pip install --no-cache \
    gradio                 \
    huggingface-hub        \
    pandas                 \
    python-dotenv

# Demo application files:
COPY ./.env             /home/reteti/.env
COPY ./reteti_core.py   /home/reteti/reteti_core.py
COPY ./reteti_text.py   /home/reteti/reteti_text.py
COPY ./demo_searcher.py /home/reteti/demo_searcher.py

# Start the demo application by default:
EXPOSE 7860
CMD ["python", "/home/reteti/demo_searcher.py"]

# docker build -t reteti-demo .
# docker buildx build -t reteti-demo .
