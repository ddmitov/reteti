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

            text_list = []
            token_list = []

            # Iterate over all texts in a batch:
            for text in batch:
                text_table_item = {}

                text_table_item['text_id'] = int(text['text_id'])
                text_table_item['text']    = str(text['text'])

                for metadata_column_name in metadata_column_names:
                    text_table_item[metadata_column_name] = \
                        text[metadata_column_name]

                # Prepare list of dictionaries:
                text_list.append(text_table_item)

                # Tokenize every text:
                tokenized_text = tokenizer.encode(
                    sequence=str(text['text']),
                    add_special_tokens=False
                )

                token_ids = tokenized_text.ids

                # Calculate token frequencies and store them in a dictionary:
                token_frequencies = {
                    token_id: token_ids.count(token_id)
                    for token_id in set(token_ids)
                }

                for token_id, token_frequency in token_frequencies.items():
                    token_item = {}

                    token_item['token']     = int(token_id)
                    token_item['text_id']   = int(text['text_id'])
                    token_item['frequency'] = int(token_frequency)

                    token_list.append(token_item)

            # Add data to the text dataset:
            texts_dataframe = pd.DataFrame(text_list)

            texts_arrow_table = duckdb.sql(
                f'''
                    SELECT
                        text_id AS partition,
                        text_id,
                        text,
                        * EXCLUDE (text_id, text)
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
            tokens_dataframe = pd.DataFrame(token_list)

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


def reteti_searcher(token_list: list) -> tuple[dict, dict]:
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

    # Step 2 - get the top N text IDs:
    step_02_start_time = time.time()

    top_results_arrow_table = duckdb.sql(
        f'''
            SELECT
                text_id,
                COUNT(DISTINCT(token)) AS unique_matching_tokens,
                SUM(frequency) AS total_matching_tokens
            FROM tokens_arrow_table tat
            GROUP BY text_id
            HAVING
                unique_matching_tokens = {len(token_set)}
                AND total_matching_tokens >= {len(token_list)}
            ORDER BY total_matching_tokens DESC
            LIMIT 10
        '''
    ).arrow()

    top_texts_list = duckdb.query(
        f'''
            SELECT text_id
            FROM top_results_arrow_table
        '''
    ).fetch_arrow_table().to_pandas()['text_id'].to_list()

    step_02_time = round((time.time() - step_02_start_time), 3)

    # Step 3 - get the top N texts:
    step_03_start_time = time.time()

    text_parquet_paths = []

    for text_id in top_texts_list:
        text_parquet_path = f'{bucket}/texts/{text_id}/part-0.parquet'
        text_parquet_paths.append(text_parquet_path)

    texts_arrow_table = pq.ParquetDataset(
        text_parquet_paths,
        filesystem=parquet_dataset_filesystem
    ).read()

    step_03_time = round((time.time() - step_03_start_time), 3)

    # Step 4 - get the final results:
    step_04_start_time = time.time()

    step_01_label = 'Step 1 - read token data ........ runtime in seconds:'
    step_02_label = 'Step 2 - get the top N text IDs . runtime in seconds:'
    step_03_label = 'Step 3 - get the top N texts .... runtime in seconds:'
    step_04_label = 'Step 4 - get the final results .. runtime in seconds:'
    total_label   = 'Total ........................... runtime in seconds:'

    try:
        search_result_dataframe = duckdb.query(
            '''
                SELECT
                    CAST(tr.total_matching_tokens AS INT) AS matching_tokens,
                    t.text_id,
                    t.* EXCLUDE (text_id, text),
                    t.text
                FROM
                    top_results_arrow_table tr
                    LEFT JOIN texts_arrow_table t ON
                        t.text_id = tr.text_id
                ORDER BY tr.total_matching_tokens DESC
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
