#!/usr/bin/env python3

# Core modules:
from   datetime import datetime
from   datetime import timedelta
import logging
from   time   import time
from   typing import List

# PIP modules:
import duckdb
import pandas          as pd
import pyarrow         as pa
import pyarrow.fs      as fs
import pyarrow.parquet as pq
from   tokenizers      import Tokenizer


def reteti_logger_starter() -> logging.Logger:
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


def reteti_indexer(
    dataset_filesystem:    fs.S3FileSystem,
    bucket:                str,
    batches_total:         int,
    batch_number:          int,
    batch:                 List[dict],
    metadata_column_names: list
) -> True:
    # Start logging: 
    logger = reteti_logger_starter()

    # Initialize tokenizer:
    tokenizer = Tokenizer.from_file('/tokenizer/tokenizer.json')

    # Start measuring time:
    processing_start = time()

    # List of dictionaries:
    token_table_list = []

    # Iterate over all texts in a batch:
    for text in batch:
        # Tokenize:
        tokenized_text = tokenizer.encode(
            sequence           = str(text['text']),
            add_special_tokens = False
        )

        token_list = tokenized_text.ids

        # Count the total number of tokens in a text:
        tokens_number = len(token_list)

        # Create token positions dictionary:
        token_positions = {}

        for index, token_id in enumerate(token_list):
            if token_id not in token_positions:
                token_positions[token_id] = []

            token_positions[token_id].append(index)

        # Create token occurences dictionary:
        token_occurences = {
            token_id: token_list.count(token_id)
            for token_id in set(token_list)
        }

        for token_id, token_occurrences in token_occurences.items():
            # Calculate a single token frequency per text:
            single_token_frequency = round((1 / len(token_list)), 5)

            token_item = {}

            token_item['token']                  = int(token_id)
            token_item['text_id']                = int(text['text_id'])
            token_item['occurrences']            = int(token_occurrences)
            token_item['positions']              = token_positions[token_id]
            token_item['single_token_frequency'] = single_token_frequency

            token_table_list.append(token_item)

    # Add data to the tokens dataset:
    tokens_dataframe = pd.DataFrame(token_table_list)

    token_arrow_table = duckdb.sql(
        f'''
            SELECT
                token AS partition,
                token,
                text_id,
                occurrences,
                positions,
                single_token_frequency
            FROM tokens_dataframe
            ORDER BY token ASC
        '''
    ).arrow()

    unique_tokens_number = duckdb.query(
        f'''
            SELECT COUNT(DISTINCT token) AS tokens
            FROM token_arrow_table
        '''
    ).fetch_arrow_table().to_pandas()['tokens'].iloc[0]

    diff_string = datetime.now().strftime('%Y-%m-%d--%H-%M-%S').strip()

    pq.write_to_dataset(
        token_arrow_table,
        filesystem             = dataset_filesystem,
        root_path              = f'{bucket}/tokens',
        partitioning           = ['partition'],
        basename_template      = 'part-{{i}}--{}.parquet'.format(diff_string),
        existing_data_behavior = 'overwrite_or_ignore',
        max_partitions         = int(unique_tokens_number)
    )

    # Calculate, display and log processing time:
    processing_time        = round((time() - processing_start), 3)
    processing_time_string = str(timedelta(seconds=processing_time))

    message = (
        f'Token batch {str(batch_number)}/{str(batches_total)} ' +
        f'processed for {processing_time_string}'
    )

    print(message, flush=True)
    logger.info(message)

    return True


def reteti_token_reader(
    bucket:             str,
    dataset_filesystem: fs.S3FileSystem,
    token:              int,
    token_occurrences:  int
) -> pa.Table:
    # If a token does not exist in the dataset,
    # this must not fail the application:
    try:
        token_arrow_table = pq.ParquetDataset(
            f'{bucket}/tokens/{token}/',
            filesystem = dataset_filesystem,
            filters    = [('occurrences', '>=', token_occurrences)]
        ).read()

        return token_arrow_table
    except FileNotFoundError:
        pass


def reteti_searcher(
    dataset_filesystem: fs.S3FileSystem,
    bucket:             str,
    tokenizer:          object,
    search_request:     str,
    results_number:     int,
    thread_pool:        object
) -> tuple[dict, dict]:
    # Tokenize user input
    token_list = tokenizer.encode(
        sequence           = search_request,
        add_special_tokens = False
    ).ids

    # Request tokens are processed here as a set to avoid
    # reading the data of the repeated tokens multiple times:
    token_set = set(token_list)

    token_data_reader_arguments = [
        (bucket, dataset_filesystem, token, token_list.count(token))
        for token in token_set
    ]

    token_result = thread_pool.starmap_async(
        reteti_token_reader,
        token_data_reader_arguments
    )

    token_arrow_tables_list = token_result.get()
    token_arrow_table = pa.concat_tables(token_arrow_tables_list)

    text_id_arrow_table = duckdb.sql(
        f'''
            WITH
                full_token_set AS (
                    SELECT text_id
                    FROM token_arrow_table
                    GROUP BY text_id
                    HAVING COUNT(DISTINCT(token)) = {str(len(token_set))}
                ),

                positions AS (
                    SELECT
                        tat.text_id,
                        tat.token,
                        tat.single_token_frequency,
                        UNNEST(tat.positions) AS position
                    FROM
                        token_arrow_table AS tat
                        INNER JOIN full_token_set AS fts
                            ON fts.text_id = tat.text_id
                ),

                distances AS (
                    SELECT
                        text_id,
                        token,
                        single_token_frequency,
                        position - LAG(position) OVER(
                            PARTITION BY text_id
                            ORDER BY position ASC
                        ) AS distance_to_previous,
                        LEAD(position) OVER(
                            PARTITION BY text_id
                            ORDER BY position ASC
                        ) - position AS distance_to_next,
                    FROM positions
                )

            SELECT
                text_id,
                ROUND(SUM(single_token_frequency), 5) AS mtf,
                FIRST(single_token_frequency) AS stf,
                COUNT(token) AS mt
            FROM distances
            WHERE
                distance_to_previous = 1
                OR distance_to_next = 1
            GROUP BY text_id
            ORDER BY mtf DESC
            LIMIT {str(results_number)}
        '''
    ).arrow()

    return text_id_arrow_table


def reteti_text_writer(
    dataset_filesystem:    fs.S3FileSystem,
    bucket:                str,
    batches_total:         int,
    batch_number:          int,
    batch:                 List[dict],
    metadata_column_names: list
) -> True:
    # Start logging: 
    logger = reteti_logger_starter()

    # Start measuring time:
    processing_start = time()

    # List of dictionaries:
    text_table_list = []

    # Iterate over all texts in a batch:
    for text in batch:
        # Prepare data for the text dataset:
        text_item = {}

        text_item['text_id'] = int(text['text_id'])
        text_item['text']    = str(text['text'])

        # Add user-created metadata column:
        for metadata_column_name in metadata_column_names:
            text_item[metadata_column_name] = text[metadata_column_name]

        text_table_list.append(text_item)

    # Add data to the text dataset.
    # Any user-created metadata column are also added.
    text_dataframe = pd.DataFrame(text_table_list)

    partition_size = 100

    text_arrow_table = duckdb.sql(
        f'''
            SELECT
                CONCAT(
                    LPAD(
                        CAST(
                            (TEXT_ID // {str(partition_size)})
                            *
                            {str(partition_size)}
                            AS VARCHAR
                        ),
                        9, '0'
                    ),
                    '-',
                    LPAD(
                        CAST(
                            (TEXT_ID // {str(partition_size)})
                            *
                            {str(partition_size)}
                            +
                            {str(partition_size - 1)}
                            AS VARCHAR
                        ),
                        9, '0'
                    )
                ) AS partition,
                text_id,
                * EXCLUDE (text_id, text),
                text
            FROM text_dataframe
        '''
    ).arrow()

    text_id_total = duckdb.query(
        f'''
            SELECT COUNT(text_id) AS text_id_total
            FROM text_arrow_table
        '''
    ).fetch_arrow_table().to_pandas()['text_id_total'].iloc[0]

    max_partitions_number = int(text_id_total) / partition_size + 1

    diff_string = datetime.now().strftime('%Y-%m-%d--%H-%M-%S').strip()

    pq.write_to_dataset(
        text_arrow_table,
        filesystem             = dataset_filesystem,
        root_path              = f'{bucket}/texts',
        partitioning           = ['partition'],
        basename_template      = 'part-{{i}}--{}.parquet'.format(diff_string),
        existing_data_behavior = 'overwrite_or_ignore',
        max_partitions         = max_partitions_number
    )

    # Calculate, display and log processing time:
    processing_time        = round((time() - processing_start), 3)
    processing_time_string = str(timedelta(seconds=processing_time))

    message = (
        f'Text batch {str(batch_number)}/{str(batches_total)} ' +
        f'written for {processing_time_string}'
    )

    print(message, flush=True)
    logger.info(message)

    return True


def reteti_text_reader(
    dataset_filesystem: fs.S3FileSystem,
    text_path:          str,
    text_id:            str
) -> pa.Table:
    # If a document does not exist in the dataset,
    # this must not fail the application:
    try:
        text_arrow_table = pq.ParquetDataset(
            text_path,
            filesystem = dataset_filesystem,
            filters    = [('text_id', '==', text_id)]
        ).read()

        return text_arrow_table
    except FileNotFoundError:
        pass


def reteti_text_extractor(
    dataset_filesystem:  fs.S3FileSystem,
    bucket:              str,
    text_id_arrow_table: pa.Table,
    thread_pool:         object
) -> pd.DataFrame:
    text_id_list = text_id_arrow_table.to_pandas()['text_id'].to_list()

    text_paths = []

    for text_id in text_id_list:
        partition_size = 100
        
        partition_start = int(text_id) // partition_size * partition_size
        partition_end   = partition_start + partition_size - 1

        partition_start_string = str(partition_start).rjust(9, '0')
        partition_end_string   = str(partition_end).rjust(9, '0')
        
        partition = f'{partition_start_string}-{partition_end_string}'

        text_path = f'{bucket}/texts/{partition}/'
        text_paths.append(text_path)

    text_arrow_table = None

    text_reader_arguments = [
        (dataset_filesystem, text_path, text_id)
        for text_path, text_id in zip(text_paths, text_id_list)
    ]

    text_result = thread_pool.starmap_async(
        reteti_text_reader,
        text_reader_arguments
    )

    text_arrow_tables_list = text_result.get()

    if len(text_arrow_tables_list) > 0:
        text_arrow_table = pa.concat_tables(text_arrow_tables_list)

    search_result_dataframe = None

    if text_id_arrow_table is not None and text_arrow_table is not None:
        search_result_dataframe = duckdb.query(
            f'''
                SELECT
                    tiat.mtf AS matching_tokens_frequency,
                    tiat.stf AS single_token_frequency,
                    tiat.mt  AS matching_tokens,
                    tat.* EXCLUDE (text),
                    tat.text
                FROM
                    text_arrow_table AS tat
                    LEFT JOIN text_id_arrow_table AS tiat
                        ON tiat.text_id = tat.text_id
                ORDER BY matching_tokens_frequency DESC
            '''
        ).fetch_arrow_table().to_pandas()

    return search_result_dataframe
