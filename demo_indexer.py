#!/usr/bin/env python3

# Python core modules:
from   datetime import datetime
from   datetime import timedelta
import gc
import logging
import os
import shutil
from   time     import time
from   typing   import List

# Python PIP modules:
from   datasets        import load_dataset
from   dotenv          import find_dotenv
from   dotenv          import load_dotenv
import duckdb
from   minio           import Minio
import pyarrow         as     pa
import pyarrow.dataset as     ds
import pyarrow.fs      as     fs

# Reteti core module:
from reteti_core import reteti_indexer
from reteti_core import reteti_file_uploader
from reteti_core import reteti_index_compactor

# Reteti supplementary module:
from reteti_text import reteti_text_writer

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
    FIRST_TABLE_NUMBER = 1
    LAST_TABLE_NUMBER  = 60
    TEXTS_PER_BATCH    = 25000

    script_start = time()
    logger = logger_starter()

    # Tigris object storage client:
    tigris_client = Minio(
        os.environ['TIGRIS_ENDPOINT_S3'],
        access_key = os.environ['TIGRIS_ACCESS_KEY_ID'],
        secret_key = os.environ['TIGRIS_SECRET_ACCESS_KEY'],
        secure     = True
    )

    # Process the dataset:
    processing_start = time()

    dataset = load_dataset(
        path  = 'stanford-oval/ccnews',
        split = 'train',
        name  = '2016'
    )

    dataset = dataset.with_format('arrow')

    processing_time = round((time() - processing_start))
    processing_time_string = str(timedelta(seconds=processing_time))

    message = f'Dataset processed for {processing_time_string}'

    print(message, flush=True)
    logger.info(message)

    # Set DuckDB sequence:
    sequence_start = 1

    if FIRST_TABLE_NUMBER > 1:
        sequence_start = ((FIRST_TABLE_NUMBER - 1) * TEXTS_PER_BATCH) + 1

    duckdb.sql(f'CREATE SEQUENCE text_id_maker START {str(sequence_start)}')

    # Iterate all text batches:
    table_number = 0
    texts_total  = 0

    for raw_table in dataset.iter(batch_size=TEXTS_PER_BATCH):
        table_number += 1

        if (
            table_number >= FIRST_TABLE_NUMBER and
            table_number <= LAST_TABLE_NUMBER
        ):
            step_start = time()

            # Prepare text data:
            batch_table = duckdb.sql(
                f'''
                    SELECT
                        NEXTVAL('text_id_maker') AS text_id,
                        title,
                        published_date AS date,
                        plain_text AS text
                    FROM raw_table
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
                f'Batch {str(table_number)}/{str(LAST_TABLE_NUMBER)} - ' +
                f'total texts: {str(batch_texts_total)}'
            )

            print(message, flush=True)
            logger.info(message)

            texts_total += batch_texts_total

            # Write text files:
            processing_start = time()

            reteti_text_writer(batch_table, '/app/data/texts')

            processing_time = round((time() - processing_start))
            processing_time_string = str(timedelta(seconds=processing_time))

            message = (
                f'Batch {str(table_number)}/{str(LAST_TABLE_NUMBER)} - ' +
                f'text files written for {processing_time_string}'
            )

            print(message, flush=True)
            logger.info(message)

            # Upload text files:
            processing_start = time()

            reteti_file_uploader(
                tigris_client,
                os.environ['TEXTS_BUCKET'],
                'texts',
                '/app/data/texts',
                'arrow',
                f'{str(table_number)}/{str(LAST_TABLE_NUMBER)}'
            )

            processing_time = round((time() - processing_start))
            processing_time_string = str(timedelta(seconds=processing_time))

            message = (
                f'Batch {str(table_number)}/{str(LAST_TABLE_NUMBER)} - ' +
                f'text files uploaded for {processing_time_string}'
            )

            print(message, flush=True)
            logger.info(message)

            # Remove local text files:
            processing_start = time()

            shutil.rmtree('/app/data/texts', ignore_errors=True)

            processing_time = round((time() - processing_start))
            processing_time_string = str(timedelta(seconds=processing_time))

            message = (
                f'Batch {str(table_number)}/{str(LAST_TABLE_NUMBER)} - ' +
                f'text files removed for {processing_time_string}'
            )

            print(message, flush=True)
            logger.info(message)

            # Write index files:
            processing_start = time()

            batch_table.drop_columns(['title', 'date'])

            reteti_indexer(
                batch_table,
                '/app/data/hashes',
                '/app/data/metadata'
            )

            processing_time = round((time() - processing_start))
            processing_time_string = str(timedelta(seconds=processing_time))

            message = (
                f'Batch {str(table_number)}/{str(LAST_TABLE_NUMBER)} - ' +
                f'index files created for {processing_time_string}'
            )

            print(message, flush=True)
            logger.info(message)

            # Calculate and log the runtime of the current batch:
            step_time = round((time() - step_start))
            step_time_string = str(timedelta(seconds=step_time))

            message = (
                f'Batch {str(table_number)}/{str(LAST_TABLE_NUMBER)} ' +
                f'processed for {step_time_string}'
            )

            print(message, flush=True)
            logger.info(message)

            # Garbage collection:
            gc.collect()

    # Compact the local index files:
    processing_start = time()

    reteti_index_compactor(
        fs.LocalFileSystem(),
        '/app/data',
        'hashes',
        'metadata'
    )

    processing_time = round((time() - processing_start))
    processing_time_string = str(timedelta(seconds=processing_time))

    message = f'Local index compacted for {processing_time_string}.'
    print(message, flush=True)
    logger.info(message)

    # Upload local index files:
    processing_start = time()

    reteti_file_uploader(
        tigris_client,
        os.environ['INDEX_BUCKET'],
        'hashes',
        '/app/data/hashes',
        'parquet',
        ''
    )

    processing_time = round((time() - processing_start))
    processing_time_string = str(timedelta(seconds=processing_time))
    message = (f'Index files uploaded for {processing_time_string}')

    print(message, flush=True)
    logger.info(message)

    # Upload local metadata file:
    processing_start = time()

    reteti_file_uploader(
        tigris_client,
        os.environ['INDEX_BUCKET'],
        'metadata',
        '/app/data/metadata',
        'parquet',
        ''
    )

    processing_time = round((time() - processing_start))
    processing_time_string = str(timedelta(seconds=processing_time))
    message = (f'Metadata file uploaded for {processing_time_string}')

    print(message, flush=True)
    logger.info(message)

    # Remove local index files:
    processing_start = time()

    shutil.rmtree('/app/data/hashes', ignore_errors=True)

    processing_time = round((time() - processing_start))
    processing_time_string = str(timedelta(seconds=processing_time))
    message = (f'Index files removed for {processing_time_string}')

    print(message, flush=True)
    logger.info(message)

    # Remove local metadata file:
    processing_start = time()

    shutil.rmtree('/app/data/metadata', ignore_errors=True)

    processing_time = round((time() - processing_start))
    processing_time_string = str(timedelta(seconds=processing_time))
    message = (f'Metadata file removed for {processing_time_string}')

    print(message, flush=True)
    logger.info(message)

    # Compact the object storage index files:
    processing_start = time()

    object_storage_filesystem = fs.S3FileSystem(
        endpoint_override = os.environ['TIGRIS_ENDPOINT_S3'],
        access_key        = os.environ['TIGRIS_ACCESS_KEY_ID'],
        secret_key        = os.environ['TIGRIS_SECRET_ACCESS_KEY'],
        scheme            = 'https'
    )

    reteti_index_compactor(
        object_storage_filesystem,
        os.environ['INDEX_BUCKET'],
        'hashes',
        'metadata'
    )

    processing_time = round((time() - processing_start))
    processing_time_string = str(timedelta(seconds=processing_time))

    message = f'Object storage index compacted for {processing_time_string}.'
    print(message, flush=True)
    logger.info(message)

    # Get script runtime:
    script_time = round((time() - script_start))
    script_time_string = str(timedelta(seconds=script_time))

    message = f'All batches processed for {script_time_string}.'
    print(message, flush=True)
    logger.info(message)

    # Log the total number of texts:
    message = f'Total texts: {str(texts_total)}'

    print(message, flush=True)
    logger.info(message)

    return True


if __name__ == '__main__':
    main()
