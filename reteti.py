#!/usr/bin/env python3

import datetime
import logging
import os
import time

from dotenv import find_dotenv
from dotenv import load_dotenv
import duckdb
import pandas          as pd
import pyarrow         as pa
import pyarrow.compute as pc
import pyarrow.fs      as fs
import pyarrow.parquet as pq
from tokenizers import Tokenizer

load_dotenv(find_dotenv())


def reteti_logger_starter() -> logging.Logger:
    start_datetime_string = (
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )

    logging.basicConfig(
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        filename=f'/app/data/logs/reteti_indexer_{start_datetime_string}.log',
        filemode='a'
    )

    logger = logging.getLogger()

    return logger


def reteti_batch_indexer(
    text_batch_list:       list,
    metadata_column_names: list
) -> True:
    # Object storage settings:
    bucket = os.environ['BUCKET']

    parquet_dataset_filesystem = fs.S3FileSystem(
        endpoint_override=os.environ['ENDPOINT_S3'],
        access_key=os.environ['ACCESS_KEY_ID'],
        secret_key=os.environ['SECRET_ACCESS_KEY'],
        scheme='http'
    )

    # Start logging: 
    logger = reteti_logger_starter()
    total_time = 0

    # Initialize tokenizer:
    tokenizer = Tokenizer.from_file('/tokenizer/tokenizer.json')

    try:
        batch_number = 0

        # Iterate over all batches of texts:
        for batch in text_batch_list:
            batch_indexing_start = time.time()

            batch_number += 1

            # Lists of dictionaries:
            text_table_list = []
            token_table_list = []

            # Iterate over all texts in a batch:
            for text in batch:
                # Tokenize:
                tokenized_text = tokenizer.encode(
                    sequence=str(text['text']),
                    add_special_tokens=False
                )

                token_list = tokenized_text.ids

                # Create a token sequence string for
                # exact match search requests:
                token_sequence_string = \
                    '|' + '|'.join(map(str, token_list)) + '|'

                # Count the total number of tokens in a text:
                tokens_number = len(token_list)

                # Prepare data for the text dataset:
                text_item = {}

                text_item['text_id']        = int(text['text_id'])
                text_item['text']           = str(text['text'])
                text_item['token_sequence'] = token_sequence_string
                text_item['tokens_number']  = tokens_number

                # Add user-created metadata column:
                for metadata_column_name in metadata_column_names:
                    text_item[metadata_column_name] = \
                        text[metadata_column_name]

                text_table_list.append(text_item)

                # Calculate token frequencies and store them in a dictionary:
                token_frequencies = {
                    token_id: token_list.count(token_id)
                    for token_id in set(token_list)
                }

                for token_id, token_frequency in token_frequencies.items():
                    token_item = {}

                    token_item['token']     = int(token_id)
                    token_item['text_id']   = int(text['text_id'])
                    token_item['frequency'] = int(token_frequency)

                    token_table_list.append(token_item)

            # Add data to the text dataset.
            # Any user-created metadata column are also added.
            texts_dataframe = pd.DataFrame(text_table_list)

            texts_arrow_table = duckdb.sql(
                f'''
                    SELECT
                        text_id AS partition,
                        text_id,
                        token_sequence,
                        tokens_number,
                        text,
                        * EXCLUDE (text_id, token_sequence, tokens_number, text)
                    FROM texts_dataframe
                '''
            ).arrow()

            unique_text_ids = duckdb.query(
                f'''
                    SELECT COUNT(text_id) AS text_ids
                    FROM texts_arrow_table
                '''
            ).fetch_arrow_table().to_pandas()['text_ids'].iloc[0]

            pq.write_to_dataset(
                texts_arrow_table,
                filesystem=parquet_dataset_filesystem,
                root_path=f'{bucket}/texts',
                partitioning=['partition'],
                basename_template='part-{i}.parquet',
                existing_data_behavior='overwrite_or_ignore',
                max_partitions=int(unique_text_ids)
            )

            # Add data to the tokens dataset:
            tokens_dataframe = pd.DataFrame(token_table_list)

            tokens_arrow_table = duckdb.sql(
                f'''
                    SELECT
                        token AS partition,
                        token,
                        text_id,
                        frequency
                    FROM tokens_dataframe
                    ORDER BY token ASC
                '''
            ).arrow()

            unique_tokens = duckdb.query(
                f'''
                    SELECT COUNT(DISTINCT token) AS tokens
                    FROM tokens_arrow_table
                '''
            ).fetch_arrow_table().to_pandas()['tokens'].iloc[0]

            date_time = datetime.datetime.now()
            date_time_string = \
                date_time.strftime('%Y-%m-%d--%H-%M-%S').strip()

            pq.write_to_dataset(
                tokens_arrow_table,
                filesystem=parquet_dataset_filesystem,
                root_path=f'{bucket}/tokens',
                partitioning=['partition'],
                basename_template='part-{{i}}--{}.parquet'.format(date_time_string),
                existing_data_behavior='overwrite_or_ignore',
                max_partitions=int(unique_tokens)
            )

            # Calculate, display and log processing times:
            batch_indexing_end = time.time()

            batch_indexing_time = round(
                (batch_indexing_end - batch_indexing_start),
                3
            )

            batch_indexing_time_string = str(
                datetime.timedelta(seconds=batch_indexing_time)
            )

            total_time = round((total_time + batch_indexing_time), 3)
            total_time_string = str(datetime.timedelta(seconds=total_time))

            print(
                f'batch {batch_number}/{len(text_batch_list)} ' +
                f'tokenized for {batch_indexing_time_string}'
            )

            logger.info(
                f'batch {batch_number}/{len(text_batch_list)} ' +
                f'tokenized for {batch_indexing_time_string}'
            )

        print('')
        print(f'total time: {total_time_string}')
        print('')

        logger.info(f'total time: {total_time_string}')
    except (KeyboardInterrupt, SystemExit):
        print('\n')
        exit(0)

    return True


def threaded_table_reader(
    parquet_dataset_filesystem: fs.S3FileSystem,
    token:                      int,
    token_frequency:            int
) -> pa.Table:
    bucket = os.environ['BUCKET']

    # If a token does not exist in the tokens dataset,
    # this must not fail the script:
    try:
        token_arrow_table = pq.ParquetDataset(
            f'{bucket}/tokens/{token}/',
            filesystem=parquet_dataset_filesystem,
            filters=[('token', '>=', token_frequency)]
        ).read()

        return token_arrow_table
    except Exception:
        pass


def reteti_searcher(
    token_list:     list,
    search_type:    str,
    results_number: int,
    thread_pool:    object
) -> tuple[dict, dict]:
    # Search criteria:

    # 1. Only texts having token frequency equal or higher than
    #    the token frequency of the search request.
    # 2. Only texts having the full set of unique tokens
    #    presented in the search request.

    # Ranking criterion - matching tokens frequency

    # Matching tokens frequency is
    # the number of matching tokens in a document
    # divided by
    # the number of all tokens in the same document.

    # Only the top N documents having
    # the highest matching tokens frequency are selected.

    # Object storage settings:
    bucket = os.environ['BUCKET']

    parquet_dataset_filesystem = fs.S3FileSystem(
        endpoint_override=os.environ['ENDPOINT_S3'],
        access_key=os.environ['ACCESS_KEY_ID'],
        secret_key=os.environ['SECRET_ACCESS_KEY'],
        scheme='http'
    )

    # Step 1 - token extraction:
    token_extraction_start_time = time.time()

    # Request tokens are processed here as a set because
    # if any token is repeated in the search request,
    # the respective token data in the dataset is read only once:
    token_set = set(token_list)

    table_reader_arguments = [
        (parquet_dataset_filesystem, token, token_list.count(token))
        for token in token_set
    ]

    result = thread_pool.starmap_async(
        threaded_table_reader,
        table_reader_arguments
    )

    token_arrow_tables_list = result.get()
    tokens_arrow_table = pa.concat_tables(token_arrow_tables_list)

    token_extraction_time = round(
        (time.time() - token_extraction_start_time),
        3
    )

    # Step 2 - token processing:
    token_processing_start_time = time.time()

    token_search_limit = ''

    if search_type == 'Approximate Match':
        token_search_limit = str(results_number)

    if search_type == 'Exact Match':
        token_search_limit = str(results_number * 10)

    text_id_arrow_table = duckdb.sql(
        f'''
            SELECT
                text_id,
                COUNT(DISTINCT(token)) AS unique_tokens,
                SUM(frequency) AS matching_tokens
            FROM tokens_arrow_table
            GROUP BY text_id
            HAVING unique_tokens = {len(token_set)}
            ORDER BY matching_tokens DESC
            LIMIT {token_search_limit}
        '''
    ).arrow()

    token_processing_time = round(
        (time.time() - token_processing_start_time),
        3
    )

    # Step 3 - document search:
    document_search_start_time = time.time()

    text_id_list = duckdb.query(
        f'''
            SELECT text_id
            FROM text_id_arrow_table
        '''
    ).fetch_arrow_table().to_pandas()['text_id'].to_list()

    text_parquet_paths = []

    for text_id in text_id_list:
        text_parquet_path = f'{bucket}/texts/{text_id}/part-0.parquet'
        text_parquet_paths.append(text_parquet_path)

    texts_arrow_table = None

    if search_type == 'Approximate Match':
        texts_arrow_table = pq.ParquetDataset(
            text_parquet_paths,
            filesystem=parquet_dataset_filesystem,
        ).read()

    if search_type == 'Exact Match':
        token_sequence_string = '|'.join(map(str, token_list))

        texts_arrow_table = pq.ParquetDataset(
            text_parquet_paths,
            filesystem=parquet_dataset_filesystem,
        ).read().filter(
            pc.match_substring(
                pc.field('token_sequence'),
                token_sequence_string
            )
        )

    search_result_dataframe = None

    try:
        search_result_dataframe = duckdb.query(
            f'''
                SELECT
                    CAST(tiat.matching_tokens AS INT) AS matching_tokens,
                    CAST(tat.tokens_number AS INT) AS total_tokens,
                    ROUND(
                        (matching_tokens / total_tokens),
                        5
                    ) AS matching_tokens_frequency,
                    tat.* EXCLUDE (
                        token_sequence,
                        tokens_number,
                        text
                    ),
                    tat.text AS text
                FROM
                    text_id_arrow_table tiat
                    LEFT JOIN texts_arrow_table tat ON
                        tat.text_id = tiat.text_id
                WHERE tat.text IS NOT NULL
                ORDER BY matching_tokens_frequency DESC
                LIMIT {str(results_number)}
            '''
        ).fetch_arrow_table().to_pandas()
    except Exception as exception:
        print(exception)
        pass

    search_result = None

    if search_result_dataframe is None:
        search_result = {}
        search_result['Message:'] = 'No matching texts were found.'

    # The results dataframe is converted to
    # a numbered list of dictionaries with numbers starting from 1:
    if search_result_dataframe is not None:
        search_result_index = range(1, len(search_result_dataframe) + 1)
        search_result_list = search_result_dataframe.to_dict('records')

        search_result = {}

        for index, element in zip(search_result_index, search_result_list):
            search_result[str(index)] = element

    document_search_time = round((time.time() - document_search_start_time), 3)

    total_time = round(
        (token_extraction_time + token_processing_time + document_search_time),
        3
    )

    search_info = {}
    search_info['Token extraction . in seconds'] = token_extraction_time
    search_info['Token processing . in seconds'] = token_processing_time
    search_info['Document search .. in seconds'] = document_search_time
    search_info['Total runtime .... in seconds'] = total_time

    return search_info, search_result
