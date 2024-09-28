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
    text_batch_list: list,
    metadata_column_names: list
) -> True:
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
                # Tokenize every text:
                tokenized_text = tokenizer.encode(
                    sequence=str(text['text']),
                    add_special_tokens=False
                )

                token_list = tokenized_text.ids

                # Create a token sequence string for
                # exact matching of search requests:
                token_sequence_string = \
                    '|' + '|'.join(map(str, token_list)) + '|'

                # Prepare data for the text dataset:
                text_item = {}

                text_item['text_id']        = int(text['text_id'])
                text_item['text']           = str(text['text'])
                text_item['token_sequence'] = token_sequence_string

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

            # Add data to the text dataset:
            texts_dataframe = pd.DataFrame(text_table_list)

            texts_arrow_table = duckdb.sql(
                f'''
                    SELECT
                        text_id AS partition,
                        text_id,
                        token_sequence,
                        text,
                        * EXCLUDE (text_id, token_sequence, text)
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


def reteti_searcher(token_list: list, search_type: str) -> tuple[dict, dict]:
    bucket = os.environ['BUCKET']

    parquet_dataset_filesystem = fs.S3FileSystem(
        endpoint_override=os.environ['ENDPOINT_S3'],
        access_key=os.environ['ACCESS_KEY_ID'],
        secret_key=os.environ['SECRET_ACCESS_KEY'],
        scheme='http'
    )

    # Step 1 - read token data:
    step_01_start_time = time.time()

    # If any token is repeated in the search request,
    # the respective token data is being read only once:
    token_set = set(token_list)
    token_arrow_tables_list = []

    for token in token_set:
        # If a token does not exist in the tokens dataset,
        # this must not fail the extraction of the existing token data:
        try:
            token_arrow_table = pq.ParquetDataset(
                f'{bucket}/tokens/{token}/',
                filesystem=parquet_dataset_filesystem
            ).read()

            token_arrow_tables_list.append(token_arrow_table)
        except Exception:
            pass

    tokens_arrow_table = pa.concat_tables(token_arrow_tables_list)

    step_01_time = round((time.time() - step_01_start_time), 3)

    # Step 2 - get text IDs:
    step_02_start_time = time.time()

    request_token_frequencies = {
        token_id: token_list.count(token_id)
        for token_id in set(token_list)
    }

    request_token_frequency_list = []

    for token_id, token_frequency in request_token_frequencies.items():
        token_item = {}
        token_item['token']     = int(token_id)
        token_item['frequency'] = int(token_frequency)

        request_token_frequency_list.append(token_item)

    request_token_frequency_dataframe = \
        pd.DataFrame(request_token_frequency_list)

    step_02_limit = ''

    if search_type == 'Approximate Match':
        step_02_limit = 'LIMIT 10'

    text_id_list = duckdb.query(
        f'''
            WITH
            eligible_texts_cte AS (
                SELECT
                    text_id,
                    COUNT(DISTINCT(token)) AS unique_tokens
                FROM tokens_arrow_table
                GROUP BY text_id
                HAVING unique_tokens = {len(token_set)}
            )

            SELECT tat.text_id AS text_id
            FROM
                tokens_arrow_table tat
                INNER JOIN eligible_texts_cte etc ON
                    etc.text_id = tat.text_id
                INNER JOIN request_token_frequency_dataframe rtfd ON
                    rtfd.token = tat.token
                    AND rtfd.frequency >= tat.frequency
            GROUP BY tat.text_id
            {step_02_limit}
        '''
    ).fetch_arrow_table().to_pandas()['text_id'].to_list()

    step_02_time = round((time.time() - step_02_start_time), 3)

    # Step 3 - get texts:
    step_03_start_time = time.time()

    text_parquet_paths = []

    for text_id in text_id_list:
        text_parquet_path = f'{bucket}/texts/{text_id}/part-0.parquet'
        text_parquet_paths.append(text_parquet_path)

    texts_arrow_table = pq.ParquetDataset(
        text_parquet_paths,
        filesystem=parquet_dataset_filesystem
    ).read()

    step_03_time = round((time.time() - step_03_start_time), 3)

    # Step 4 - get the final results:
    step_04_start_time = time.time()

    step_01_label = 'Step 1 - read token data ... runtime in seconds:'
    step_02_label = 'Step 2 - get text IDs ...... runtime in seconds:'
    step_03_label = 'Step 3 - get texts ......... runtime in seconds:'
    step_04_label = 'Step 4 - get final results . runtime in seconds:'
    total_label   = 'Total ...................... runtime in seconds:'

    token_sequence_string = '|'.join(map(str, token_list))

    step_04_condition = ''

    if search_type == 'Exact Match':
        step_04_condition = \
            f"WHERE token_sequence LIKE '%{token_sequence_string}%'"

    try:
        search_result_dataframe = duckdb.query(
            f'''
                SELECT
                    (
                        CAST({len(token_list)} AS FLOAT)
                        /
                        CAST(LEN(STRING_SPLIT(token_sequence, '|')) AS FLOAT)
                    ) AS token_frequency,
                    text_id,
                    * EXCLUDE (text_id, token_sequence, text),
                    text
                FROM texts_arrow_table
                {step_04_condition}
                ORDER BY token_frequency DESC
                LIMIT 10
            '''
        ).fetch_arrow_table().to_pandas()
    except Exception:
        step_04_time = round((time.time() - step_04_start_time), 3)

        total_time = round(
            (step_01_time + step_02_time + step_03_time + step_04_time),
            3
        )

        search_info = {}
        search_info[step_01_label] = step_01_time
        search_info[step_02_label] = step_02_time
        search_info[step_03_label] = step_03_time
        search_info[step_04_label] = step_04_time
        search_info[total_label]   = total_time

        search_result = {}
        search_result['Message:'] = 'No matching texts were found.'

        return search_info, search_result

    search_result = search_result_dataframe.to_dict('records')

    step_04_time = round((time.time() - step_04_start_time), 3)

    total_time = round(
        (step_01_time + step_02_time + step_03_time + step_04_time),
        3
    )

    search_info = {}
    search_info[step_01_label] = step_01_time
    search_info[step_02_label] = step_02_time
    search_info[step_03_label] = step_03_time
    search_info[step_04_label] = step_04_time
    search_info[total_label]   = total_time

    return search_info, search_result
