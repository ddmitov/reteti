#!/usr/bin/env python3

# Core modules:
import datetime
import math
import os
import time

# PIP modules:
from   dotenv          import find_dotenv
from   dotenv          import load_dotenv
import duckdb
from   huggingface_hub import hf_hub_download
from   minio           import Minio
import pyarrow         as     pa
import pyarrow.fs      as     fs

# Private module:
from reteti import reteti_logger_starter
from reteti import reteti_indexer
from reteti import reteti_text_writer
from reteti import reteti_index_compactor

ROWS_PER_BATCH = 10000

load_dotenv(find_dotenv())


def dataset_filesystem_starter() -> fs.S3FileSystem:
    dataset_filesystem = fs.S3FileSystem(
        endpoint_override = os.environ['ENDPOINT_S3'],
        access_key        = os.environ['ACCESS_KEY_ID'],
        secret_key        = os.environ['SECRET_ACCESS_KEY'],
        scheme            = 'http'
    )

    return dataset_filesystem


def data_preprocessor() -> int:
    # Start logging: 
    logger = reteti_logger_starter()

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
    duckdb.sql('CREATE TEMPORARY SEQUENCE text_id_maker START 1')

    # Temporary parquet files:
    temp_file_names = []

    message = 'Processing Common Crawl News Bulgarian data.'
    print(message, flush=True)
    logger.info(message)

    # A maximum of 438 794 records can be returned without the LIMIT clause.
    bg_arrow_table = duckdb.sql(
        f'''
            SELECT
                NEXTVAL('text_id_maker') AS text_id,
                date_publish_final AS date,
                REPLACE(title, '\n', '') AS title,
                REPLACE(maintext, '\n', ' ') AS text,
            FROM read_json_auto("/app/data/hf/2021/bg.jsonl.gz")
            WHERE
                date_publish_final IS NOT NULL
                AND title IS NOT NULL
                AND title NOT LIKE '%...'
                AND maintext NOT LIKE '%...'
                AND LENGTH(maintext) <= 2000
                LIMIT 400000
        '''
    ).to_arrow_table()

    bg_batches_number = math.ceil(bg_arrow_table.num_rows // ROWS_PER_BATCH)

    for index in range(bg_batches_number):
        batch_table = bg_arrow_table.slice(
            offset = index * ROWS_PER_BATCH,
            length = ROWS_PER_BATCH
        )

        temp_file_name = f'bg_{str(index + 1)}.parquet'
        temp_file_names.append(temp_file_name)

        duckdb.sql(
            f'''
                COPY (SELECT * FROM batch_table)
                TO "/app/data/temp/{temp_file_name}"
                (FORMAT PARQUET)
            '''
        )

    del bg_arrow_table

    message = 'Processing Common Crawl News English data.'
    print(message, flush=True)
    logger.info(message)

    en_arrow_table = duckdb.sql(
        f'''
            SELECT
                NEXTVAL('text_id_maker') AS text_id,
                date_publish_final AS date,
                REPLACE(title, '\n', '') AS title,
                REPLACE(maintext, '\n', ' ') AS text,
            FROM read_json_auto("/app/data/hf/2021/en01.jsonl.gz")
            WHERE
                date_publish_final IS NOT NULL
                AND title IS NOT NULL
                AND title NOT LIKE '%...'
                AND maintext NOT LIKE '%...'
                AND LENGTH(maintext) <= 2000
                LIMIT 600000
        '''
    ).to_arrow_table()

    en_batches_number = math.ceil(en_arrow_table.num_rows // ROWS_PER_BATCH)

    for index in range(en_batches_number):
        batch_table = en_arrow_table.slice(
            offset = index * ROWS_PER_BATCH,
            length = ROWS_PER_BATCH
        )

        temp_file_name = f'en_{str(index + 1)}.parquet'
        temp_file_names.append(temp_file_name)

        duckdb.sql(
            f'''
                COPY (SELECT * FROM batch_table)
                TO "/app/data/temp/{temp_file_name}"
                (FORMAT PARQUET)
            '''
        )

    del en_arrow_table

    return temp_file_names


def main():
    # Start logging: 
    logger = reteti_logger_starter()

    # Create buckets if they don't exist:
    minio_client = Minio(
        'minio:9000',
        access_key = os.environ['ACCESS_KEY_ID'],
        secret_key = os.environ['SECRET_ACCESS_KEY'],
        secure     = False
    )

    if not minio_client.bucket_exists(os.environ['INDEX_BUCKET']):
        minio_client.make_bucket(os.environ['INDEX_BUCKET'])

    if not minio_client.bucket_exists(os.environ['INDEX_COMPACT_BUCKET']):
        minio_client.make_bucket(os.environ['INDEX_COMPACT_BUCKET'])

    if not minio_client.bucket_exists(os.environ['TEXTS_BUCKET']):
        minio_client.make_bucket(os.environ['TEXTS_BUCKET'])

    # Initialize Parquet dataset filesystem in object storage:
    dataset_filesystem = dataset_filesystem_starter()

    # Object storage buckets:
    index_bucket = os.environ['INDEX_BUCKET']
    index_compact_bucket = os.environ['INDEX_COMPACT_BUCKET']
    texts_bucket = os.environ['TEXTS_BUCKET']

    # Start measuring processing time:
    total_processing_start = time.time()

    # Pre-process input texts and group them in batches:
    temp_file_names = data_preprocessor()

    # Index all text batches:
    metadata_column_names = ['title', 'date']

    temp_file_number = 0

    for temp_file_name in temp_file_names:
        temp_file_number += 1

        texts_list = duckdb.sql(
            f'''
                SELECT
                    text_id,
                    date,
                    title,
                    text,
                FROM "/app/data/temp/{temp_file_name}"
            '''
        ).to_arrow_table().to_pylist()

        reteti_text_writer(
            dataset_filesystem,
            texts_bucket,
            len(temp_file_names),
            temp_file_number,
            texts_list,
            metadata_column_names
        )

        reteti_indexer(
            dataset_filesystem,
            index_bucket,
            len(temp_file_names),
            temp_file_number,
            texts_list,
            metadata_column_names
        )

    reteti_index_compactor(
        dataset_filesystem,
        index_bucket,
        index_compact_bucket
    )

    total_processing_time = round((time.time() - total_processing_start), 3)

    total_processing_time_string = str(
        datetime.timedelta(seconds=total_processing_time)
    )

    message = f'All texts processed for {total_processing_time_string}'

    print(message, flush=True)
    logger.info(message)

    return True


if __name__ == '__main__':
    main()
