#!/usr/bin/env python3

# Python core modules:
from   datetime import datetime
from   datetime import timedelta
from   datetime import timezone
import gc
from   io       import BytesIO
import json
import logging
import os
import shutil
from   time     import time

# Python PIP modules:
from   datasets import load_dataset
from   dotenv   import find_dotenv
from   dotenv   import load_dotenv
import duckdb
from   minio    import Minio

# Reteti core module:
from reteti_core import reteti_binned_index_writer
from reteti_core import reteti_index_formatter

# Reteti supplementary module:
from reteti_text import reteti_text_uploader

# Start the indexing process:
# docker run --rm -it --user $(id -u):$(id -g) \
# -v $PWD:/app \
# -v $PWD/data:/.cache \
# reteti-demo python /app/demo_indexer.py

load_dotenv(find_dotenv())


def logger_starter() -> logging.Logger:
    start_datetime_string = (datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    os.makedirs('/app/data/logs', exist_ok=True)

    logging.basicConfig(
        level    = logging.INFO,
        datefmt  = '%Y-%m-%d %H:%M:%S',
        format   = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        filename = f'/app/data/logs/reteti_{start_datetime_string}.log',
        filemode = 'a'
    )

    logger = logging.getLogger()

    return logger


def main():
    YEAR = '2024'

    FIRST_DATASET_TABLE_NUMBER = 151
    LAST_DATASET_TABLE_NUMBER  = 300
    TEXTS_PER_DATASET_TABLE    = 25000

    BINS_TOTAL = 500

    # Start measuring runtime and set logging:
    script_start = time()
    logger = logger_starter()

    # Set object storage client:
    object_storage_client = Minio(
        os.environ['ENDPOINT_S3'],
        access_key = os.environ['ACCESS_KEY_ID'],
        secret_key = os.environ['SECRET_ACCESS_KEY'],
        secure     = True
    )

    # Initialize a stopwords list:
    stopword_set = None

    with open('/home/reteti/stopwords-iso.json', 'r') as stopwords_json_file:
        stopword_json_data = json.load(stopwords_json_file)

        stopwords_bg = set(stopword_json_data['bg'])
        stopwords_en = set(stopword_json_data['en'])

        stopword_set = stopwords_bg | stopwords_en

    # Get the number of the already indexed texts:
    json_data = None

    previous_texts_total = 0
    generation_timestamp = datetime.now(timezone.utc)

    try:
        response = object_storage_client.get_object(
            os.environ['INDEX_BUCKET'],
            'metadata/metadata.json'
        )

        json_data = json.loads(response.read().decode('utf-8'))
    except Exception:
        pass

    try:
        previous_texts_total = json_data['texts_total']
    except Exception:
        pass

    try:
        generation_timestamp = datetime.fromisoformat(
            json_data['generation_timestamp']
        )
    except Exception:
        pass

    message = f'Texts from previous script runs: {str(previous_texts_total)}'

    print(message, flush=True)
    logger.info(message)

    # Total number of texts from the current script run:
    script_run_texts_total = 0

    # Set DuckDB sequence for the generation of text_id numbers:
    sequence_start = previous_texts_total + 1

    duckdb.sql(f'CREATE SEQUENCE text_id_maker START {str(sequence_start)}')

    # Initialize the dataset:
    print('Reading dataset ...', flush=True)

    dataset = load_dataset(
        path      = 'stanford-oval/ccnews',
        split     = 'train',
        name      = YEAR,
        streaming = True
    ).with_format('arrow')

    # Iterate the dataset:
    table_number = 0

    for dataset_table in dataset.iter(batch_size=TEXTS_PER_DATASET_TABLE):
        table_number += 1

        if (
            table_number >= FIRST_DATASET_TABLE_NUMBER and
            table_number <= LAST_DATASET_TABLE_NUMBER
        ):
            # Prepare text data:
            batch_table = duckdb.sql(
                f'''
                    SELECT
                        NEXTVAL('text_id_maker') AS text_id,
                        title,
                        published_date AS date,
                        plain_text AS text
                    FROM dataset_table
                    WHERE language IN ('bg', 'en')
                '''
            ).arrow()

            # Get the number of texts in the batch:
            batch_texts_total = duckdb.query(
                '''
                    SELECT COUNT(text_id) AS texts_total
                    FROM batch_table
                '''
            ).arrow().column('texts_total')[0].as_py()

            message = (
                f'Batch {str(table_number)}/{str(LAST_DATASET_TABLE_NUMBER)} - ' +
                f'texts: {str(batch_texts_total)}'
            )

            print(message, flush=True)
            logger.info(message)

            script_run_texts_total += batch_texts_total

            # Upload texts:
            processing_start = time()

            message_header = (
                f'text batch {str(table_number)}/{str(LAST_DATASET_TABLE_NUMBER)}'
            )

            reteti_text_uploader(
                object_storage_client,
                os.environ['TEXTS_BUCKET'],
                'texts',
                batch_table,
                message_header
            )

            processing_time = round((time() - processing_start))
            processing_time_string = str(timedelta(seconds=processing_time))

            message = (
                f'Batch {str(table_number)}/{str(LAST_DATASET_TABLE_NUMBER)} - ' +
                f'texts uploaded for {processing_time_string}'
            )

            print(message, flush=True)
            logger.info(message)

            # Write binned index:
            processing_start = time()

            batch_table.drop_columns(['title', 'date'])

            reteti_binned_index_writer(
                BINS_TOTAL,
                batch_table,
                '/app/data/binned_index',
                '/app/data/binned_index_metadata',
                stopword_set
            )

            processing_time = round((time() - processing_start))
            processing_time_string = str(timedelta(seconds=processing_time))

            message = (
                f'Batch {str(table_number)}/{str(LAST_DATASET_TABLE_NUMBER)} - ' +
                f'binned index created for {processing_time_string}'
            )

            print(message, flush=True)
            logger.info(message)

        if table_number == LAST_DATASET_TABLE_NUMBER:
            break

        # Garbage collection:
        gc.collect()

    # Format the binned index:
    processing_start = time()

    reteti_index_formatter(
        object_storage_client,
        os.environ['INDEX_BUCKET'],
        'index',
        BINS_TOTAL,
        generation_timestamp,
        '/app/data/binned_index',
        '/app/data/binned_index_metadata'
    )

    processing_time = round((time() - processing_start))
    processing_time_string = str(timedelta(seconds=processing_time))

    message = f'Index formatted for {processing_time_string}.'
    print(message, flush=True)
    logger.info(message)

    # Remove all dataset files:
    shutil.rmtree('/app/data/huggingface', ignore_errors=True)

    # Save the total texts number and the generation timestamp:
    combined_texts_total = previous_texts_total + script_run_texts_total
    generation_timestamp = datetime.now(timezone.utc)

    updated_json_data = {
        'texts_total':          combined_texts_total,
        'generation_timestamp': generation_timestamp.isoformat()
    }

    json_bytes = json.dumps(updated_json_data, indent=4).encode('utf-8')

    object_storage_client.put_object(
        bucket_name  = os.environ['INDEX_BUCKET'],
        object_name  = 'metadata/metadata.json',
        data         = BytesIO(json_bytes),
        length       = len(json_bytes),
        content_type = 'application/json'
    )

    message = f'Texts from this script run: {str(script_run_texts_total)}'

    print(message, flush=True)
    logger.info(message)

    message = f'Total texts: {str(combined_texts_total)}'

    print(message, flush=True)
    logger.info(message)

    # Get the script runtime:
    script_time = round((time() - script_start))
    script_time_string = str(timedelta(seconds=script_time))

    message = f'Total script runtime: {script_time_string}.'
    print(message, flush=True)
    logger.info(message)

    return True


if __name__ == '__main__':
    main()
