#!/usr/bin/env python3

# Python core modules:
from   datetime import datetime
from   datetime import timedelta
import logging
import os
import time
from   typing  import List

# Python PIP modules:
from   dotenv          import find_dotenv
from   dotenv          import load_dotenv
import duckdb
from   huggingface_hub import hf_hub_download
from   minio           import Minio
import pyarrow         as     pa
import pyarrow.dataset as     ds
import pyarrow.fs      as     fs

# Reteti core module:
from reteti_core import reteti_list_splitter
from reteti_core import reteti_indexer
from reteti_core import reteti_index_compactor

# Reteti supplementary modules:
from reteti_file import reteti_file_uploader
from reteti_text import reteti_text_writer

load_dotenv(find_dotenv())


def logger_starter() -> logging.Logger:
    start_datetime_string = (datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    logging.basicConfig(
        level    = logging.DEBUG,
        datefmt  = '%Y-%m-%d %H:%M:%S',
        format   = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        filename = f'/app/data/logs/reteti_{start_datetime_string}.log',
        filemode = 'a'
    )

    logger = logging.getLogger()

    return logger


def dataset_text_extractor(
    file_path: str,
    limit:     int
) -> pa.Table:
    arrow_table = duckdb.sql(
        f'''
            SELECT
                NEXTVAL('text_id_maker') AS text_id,
                date_publish_final AS date,
                REPLACE(title, '\n', '') AS title,
                REPLACE(maintext, '\n', ' ') AS text,
            FROM read_json_auto("{file_path}")
            WHERE
                date_publish_final IS NOT NULL
                AND title IS NOT NULL
                AND title NOT LIKE '%...'
                AND maintext NOT LIKE '%...'
                AND LENGTH(maintext) <= 2000
                LIMIT {str(limit)}
        '''
    ).to_arrow_table()

    return arrow_table


def dataset_text_processor(logger: object) -> list:
    # Download data from Hugging Face dataset or open a locally cached copy:
    message = 'Obtaining Common Crawl News Bulgarian data.'
    print(message, flush=True)
    logger.info(message)

    hf_hub_download(
        repo_id   = 'CloverSearch/cc-news-mutlilingual',
        filename  = '2021/bg.jsonl.gz',
        local_dir = '/app/data/hf',
        repo_type = 'dataset'
    )

    message = 'Obtaining Common Crawl News English data.'
    print(message, flush=True)
    logger.info(message)

    hf_hub_download(
        repo_id   = 'CloverSearch/cc-news-mutlilingual',
        filename  = '2021/en01.jsonl.gz',
        local_dir = '/app/data/hf',
        repo_type = 'dataset'
    )

    # Set a DuckDB sequence to produce unique text_id numbers:
    duckdb.sql('CREATE SEQUENCE text_id_maker START 1')

    ROWS_PER_BATCH = 10000
    text_file_list = []

    message = 'Processing Common Crawl News Bulgarian data.'
    print(message, flush=True)
    logger.info(message)

    bg_arrow_table = dataset_text_extractor(
        '/app/data/hf/2021/bg.jsonl.gz',
        400000
    )

    bg_batches_total = bg_arrow_table.num_rows // ROWS_PER_BATCH

    for batch_number in range(bg_batches_total):
        batch_table = bg_arrow_table.slice(
            offset = batch_number * ROWS_PER_BATCH,
            length = ROWS_PER_BATCH
        )

        batch_text_file_list = reteti_text_writer(batch_table, logger)
        text_file_list.extend(batch_text_file_list)

    del bg_arrow_table

    message = 'Processing Common Crawl News English data.'
    print(message, flush=True)
    logger.info(message)

    en_arrow_table = dataset_text_extractor(
        '/app/data/hf/2021/en01.jsonl.gz',
        600000
    )

    en_batches_total = en_arrow_table.num_rows // ROWS_PER_BATCH

    for batch_number in range(en_batches_total):
        batch_table = en_arrow_table.slice(
            offset = batch_number * ROWS_PER_BATCH,
            length = ROWS_PER_BATCH
        )

        batch_text_file_list = reteti_text_writer(batch_table, logger)
        text_file_list.extend(batch_text_file_list)

    del en_arrow_table

    return text_file_list


def main():
    logger = logger_starter()
    total_processing_start = time.time()

    text_filenames = dataset_text_processor(logger)

    partitioned_text_filenames = reteti_list_splitter(text_filenames, 100)

    batch_number = 0

    for text_batch in partitioned_text_filenames:
        batch_number += 1

        text_list = ds.dataset(
            text_batch,
            format     = 'arrow',
            filesystem = fs.LocalFileSystem()
        ).to_table().to_pylist()

        reteti_indexer(
            len(partitioned_text_filenames),
            batch_number,
            text_list,
            logger
        )

    reteti_index_compactor(
        '/app/data/reteti-index',
        '/app/data/reteti-compact-index',
        logger
    )

    # minio_local_client = Minio(
    #     'minio:9000',
    #     access_key = os.environ['LOCAL_ACCESS_KEY_ID'],
    #     secret_key = os.environ['LOCAL_SECRET_ACCESS_KEY'],
    #     secure     = False
    # )

    # if not minio_local_client.bucket_exists(os.environ['INDEX_BUCKET']):
    #     minio_local_client.make_bucket(os.environ['INDEX_BUCKET'])

    # if not minio_local_client.bucket_exists(os.environ['TEXTS_BUCKET']):
    #     minio_local_client.make_bucket(os.environ['TEXTS_BUCKET'])

    tigris_client = Minio(
        os.environ['TIGRIS_ENDPOINT_S3'],
        access_key = os.environ['TIGRIS_ACCESS_KEY_ID'],
        secret_key = os.environ['TIGRIS_SECRET_ACCESS_KEY'],
        secure     = True
    )

    reteti_file_uploader(
        tigris_client,
        os.environ['INDEX_BUCKET'],
        '/app/data/reteti-compact-index',
        'parquet'
    )

    reteti_file_uploader(
        tigris_client,
        os.environ['TEXTS_BUCKET'],
        '/app/data/reteti-texts',
        'arrow'
    )

    total_processing_time = round((time.time() - total_processing_start), 3)
    total_processing_time_string = str(timedelta(seconds=total_processing_time))

    message = f'All texts processed for {total_processing_time_string}'
    print(message, flush=True)
    logger.info(message)

    return True


if __name__ == '__main__':
    main()
