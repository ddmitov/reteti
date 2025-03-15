#!/usr/bin/env python3

# Python core modules:
from   datetime import datetime
from   datetime import timedelta
import logging
import os
from   time     import time
from   typing   import List

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
from reteti_core import reteti_dataset_producer

# Reteti supplementary modules:
from reteti_file import reteti_file_cleaner
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


def recursive_files_lister(root_dir: str) -> List[str]:
    return [
        os.path.join(root, filename)
        for root, _, filenames in os.walk(root_dir) 
        for filename in filenames
    ]


def dataset_text_extractor(
    file_path: str,
    limit:     int
) -> pa.Table:
    table = duckdb.sql(
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
    ).to_table()

    return table


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

    bg_table = dataset_text_extractor(
        '/app/data/hf/2021/bg.jsonl.gz',
        400000
    )

    bg_batches_total = bg_table.num_rows // ROWS_PER_BATCH

    for batch_number in range(bg_batches_total):
        batch_table = bg_table.slice(
            offset = batch_number * ROWS_PER_BATCH,
            length = ROWS_PER_BATCH
        )

        batch_text_file_list = reteti_text_writer(
            batch_number + 1,
            bg_batches_total,
            batch_table,
            logger
        )

        text_file_list.extend(batch_text_file_list)

    message = 'Processing Common Crawl News English data.'
    print(message, flush=True)
    logger.info(message)

    en_table = dataset_text_extractor(
        '/app/data/hf/2021/en01.jsonl.gz',
        600000
    )

    en_batches_total = en_table.num_rows // ROWS_PER_BATCH

    for batch_number in range(en_batches_total):
        batch_table = en_table.slice(
            offset = batch_number * ROWS_PER_BATCH,
            length = ROWS_PER_BATCH
        )

        batch_text_file_list = reteti_text_writer(
            batch_number + 1,
            en_batches_total,
            batch_table,
            logger
        )

        text_file_list.extend(batch_text_file_list)

    return text_file_list


def main():
    processing_start = time()
    logger = logger_starter()

    duckdb_connection = duckdb.connect(
        '/app/data/reteti/reteti.db',
        config = {'allocator_background_threads': True}
    )

    duckdb_connection.sql('INSTALL crypto FROM community')
    duckdb_connection.sql('LOAD crypto')

    duckdb_connection.sql(
        '''
            CREATE TABLE IF NOT EXISTS hash_table(
                hash        VARCHAR,
                text_id     VARCHAR,
                positions   INTEGER[],
                total_words INT
            )
        '''
    )

    # print('Extracting text data ...', flush=True)

    # # text_filenames = dataset_text_processor(logger)
    # text_filenames = recursive_files_lister('/app/data/reteti/texts')

    # print('Indexing ...', flush=True)

    # partitioned_text_filenames = reteti_list_splitter(text_filenames, 100)
    # batch_number = 0

    # for text_batch in partitioned_text_filenames:
    #     batch_number += 1

    #     text_table = ds.dataset(
    #         text_batch,
    #         format     = 'arrow',
    #         filesystem = fs.LocalFileSystem()
    #     ).to_table()

    #     text_table.drop_columns(['date', 'title'])

    #     reteti_indexer(
    #         len(partitioned_text_filenames),
    #         batch_number,
    #         text_table,
    #         duckdb_connection,
    #         logger
    #     )

    # reteti_dataset_producer('/app/data/reteti', duckdb_connection, logger)

    tigris_client = Minio(
        os.environ['TIGRIS_ENDPOINT_S3'],
        access_key = os.environ['TIGRIS_ACCESS_KEY_ID'],
        secret_key = os.environ['TIGRIS_SECRET_ACCESS_KEY'],
        secure     = True
    )

    reteti_file_uploader(
        tigris_client,
        os.environ['INDEX_BUCKET'],
        'hashes',
        '/app/data/reteti/hashes',
        'parquet'
    )

    reteti_file_uploader(
        tigris_client,
        os.environ['TEXTS_BUCKET'],
        'texts',
        '/app/data/reteti/texts',
        'arrow'
    )

    processing_time = round((time() - processing_start), 3)
    processing_time_string = str(timedelta(seconds=processing_time))

    message = f'All texts processed for {processing_time_string}'
    print(message, flush=True)
    logger.info(message)

    return True


if __name__ == '__main__':
    main()
