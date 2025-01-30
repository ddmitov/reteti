#!/usr/bin/env python3

# Python core modules:
from   datetime import datetime
from   datetime import timedelta
from   multiprocessing.pool import ThreadPool
import os
from   pathlib  import Path
import re
from   time     import time
from   typing   import List

# Python PIP modules:
import duckdb
import pandas          as pd
import pyarrow         as pa
import pyarrow.dataset as ds
import pyarrow.feather as ft
import pyarrow.fs      as fs
import pyarrow.parquet as pq
from   tokenizers      import Tokenizer


def reteti_list_splitter(input_list: list, parts_number: int) -> List[list]:
    base_size = len(input_list) // parts_number
    remainder = len(input_list) % parts_number

    result = []
    current_index = 0

    for index in range(parts_number):
        partition_size = base_size + (1 if index < remainder else 0)
        result.append(input_list[current_index:current_index + partition_size])
        current_index += partition_size

    return result


def reteti_indexer(
    batches_total: int,
    batch_number:  int,
    batch:         List[dict],
    logger:        object
) -> True:
    token_paths = []


    def tokens_file_visitor(written_file):
        token_paths.append(written_file.path)


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
        '''
            SELECT
                token AS partition,
                token,
                text_id,
                occurrences,
                positions,
                single_token_frequency
            FROM tokens_dataframe
            ORDER BY partition ASC
        '''
    ).arrow()

    unique_partitions_number = duckdb.query(
        '''
            SELECT COUNT(DISTINCT partition) AS partitions
            FROM token_arrow_table
        '''
    ).fetch_arrow_table().to_pandas()['partitions'].iloc[0]

    diff_string = datetime.now().strftime('%Y-%m-%d--%H-%M-%S').strip()

    pq.write_to_dataset(
        token_arrow_table,
        filesystem             = fs.LocalFileSystem(),
        root_path              = '/app/data/reteti-index/tokens',
        partitioning           = ['partition'],
        basename_template      = 'part-{{i}}--{}.parquet'.format(diff_string),
        existing_data_behavior = 'overwrite_or_ignore',
        max_partitions         = int(unique_partitions_number),
        file_visitor           = tokens_file_visitor
    )

    # Add data to the metadata dataset:
    token_path_dataframe = pd.DataFrame(token_paths, columns=['path'])

    token_arrow_table = duckdb.sql(
        '''
            SELECT
                path,
                REGEXP_EXTRACT(path, '\\d{1,10}') AS token,
                NOW() AS datetime
            FROM token_path_dataframe
        '''
    ).arrow()

    pq.write_to_dataset(
        token_arrow_table,
        filesystem             = fs.LocalFileSystem(),
        root_path              = '/app/data/reteti-index/metadata/paths',
        basename_template      = 'part-{{i}}--{}.parquet'.format(diff_string),
        existing_data_behavior = 'overwrite_or_ignore'
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


def reteti_index_compactor_worker(
    index_directory:         str,
    compact_index_directory: str,
    token_list:              list,
    batch_number:            int,
    logger:                  object
) -> True:
    token_number = 0

    for token in token_list:
        token_number += 1

        token_arrow_table = pq.ParquetDataset(
            f'{index_directory}/tokens/{token}/',
            filesystem = fs.LocalFileSystem()
        ).read()

        Path(f'{compact_index_directory}/tokens/{token}',).mkdir(
            parents  = True,
            exist_ok = True
        )

        pq.write_table(
            token_arrow_table,
            f'{compact_index_directory}/tokens/{token}/{token}.parquet',
            filesystem  = fs.LocalFileSystem(),
            compression = 'NONE'
        )

        message = (
            f'Batch {str(batch_number)} - ' +
            f'{str(token_number)}/{str(len(token_list))} - ' +
            f'compacted data for token {token}'
        )

        print(message, flush=True)
        logger.info(message)

    return True


def reteti_index_compactor(
    index_directory:         str,
    compact_index_directory: str,
    logger:                  object
) -> True:
    path_arrow_table = pq.ParquetDataset(
        f'{index_directory}/metadata/paths/',
        filesystem = fs.LocalFileSystem()
    ).read()

    unique_token_list = duckdb.query(
        '''
            SELECT token
            FROM path_arrow_table
            GROUP BY token
        '''
    ).fetch_arrow_table().to_pandas()['token']

    Path(f'{compact_index_directory}/tokens').mkdir(
        parents  = True,
        exist_ok = True
    )

    token_batch_list = reteti_list_splitter(unique_token_list, 4)

    thread_pool = ThreadPool(4)

    index_compactor_worker_arguments = [
        (
            index_directory,
            compact_index_directory,
            token_batch,
            batch_number,
            logger
        )
        for token_batch, batch_number in zip(
            token_batch_list,
            range(len(token_batch_list))
        )
    ]

    token_result = thread_pool.starmap_async(
        reteti_index_compactor_worker,
        index_compactor_worker_arguments
    )

    token_result_list = token_result.get()

    return True


def reteti_token_reader(
    dataset_filesystem: fs.S3FileSystem,
    token_path:         str,
    token_occurrences:  int
) -> None | pa.Table:
    token_arrow_table = None

    try:
        token_arrow_table = pq.read_table(
            token_path,
            filesystem = dataset_filesystem,
            filters    = [('occurrences', '>=', token_occurrences)]
        )
    except FileNotFoundError:
        pass

    return token_arrow_table


def reteti_searcher(
    dataset_filesystem: fs.S3FileSystem,
    bucket:             str,
    tokenizer:          object,
    search_request:     str,
    results_number:     int,
    thread_pool:        object
) -> None | pa.Table:
    # Tokenize user input
    token_list = tokenizer.encode(
        sequence           = search_request,
        add_special_tokens = False
    ).ids

    # Request tokens are processed here as a set
    # to avoid reading data of repeated tokens multiple times:
    token_set = set(token_list)

    token_reader_arguments = [
        (
            dataset_filesystem,
            f'{bucket}/tokens/{token}/{token}.parquet',
            token_list.count(token)
        )
        for token in token_set
    ]

    token_result = thread_pool.starmap_async(
        reteti_token_reader,
        token_reader_arguments
    )

    token_result_list = token_result.get()

    token_arrow_tables_list = [
        result for result in token_result_list
        if result is not None
    ]

    token_arrow_table = None

    if len(token_arrow_tables_list) < len(token_set):
        return None

    if len(token_arrow_tables_list) == len(token_set):
        token_arrow_table = pa.concat_tables(token_arrow_tables_list)

    token_sequence_string = ''.join(map(str, token_list))

    text_id_arrow_table = duckdb.sql(
        f'''
            WITH
                single_token_frequencies AS (
                    SELECT
                        text_id,
                        single_token_frequency
                    FROM token_arrow_table
                    GROUP BY
                        text_id,
                        single_token_frequency
                ),

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
                        position,
                        LEAD(
                            position,
                            {str(len(token_list) - 1)}
                        ) OVER (
                            PARTITION BY text_id
                            ORDER BY position ASC
                        ) - position
                        AS distance_to_end,
                    FROM positions
                ),

                start_positions AS (
                    SELECT
                        text_id,
                        token,
                        position,
                        CASE
                            WHEN
                                token = {str(token_list[0])}
                                AND distance_to_end = {str(len(token_list) - 1)}
                            THEN position
                            ELSE NULL
                        END AS start_position
                    FROM distances
                ),

                sequences AS (
                    SELECT
                        text_id,
                        token,
                        position,
                        FIRST(start_position) OVER (
                            PARTITION BY text_id
                            ORDER BY position
                            ROWS BETWEEN
                                {str(len(token_list) - 1)} PRECEDING
                                AND CURRENT ROW
                        ) AS sequence_id
                    FROM start_positions
                ),

                sequences_aggregated AS (
                    SELECT
                        text_id,
                        sequence_id,
                        STRING_AGG(token, '' ORDER BY position) AS sequence
                    FROM sequences
                    WHERE sequence_id IS NOT NULL
                    GROUP BY
                        text_id,
                        sequence_id
                    HAVING
                        COUNT(token) = {str(len(token_list))}
                        AND FIRST(token ORDER BY position) = {str(token_list[0])}
                        AND LAST(token ORDER BY position) = {str(token_list[-1])}
                        AND sequence = '{token_sequence_string}'
                )

            SELECT
                s.text_id,
                COUNT(s.sequence_id) AS hits,
                hits * {str(len(token_list))} AS matching_tokens,
                FIRST(stf.single_token_frequency) AS single_token_frequency,
                ROUND(
                    (FIRST(stf.single_token_frequency) * matching_tokens),
                    5
                ) AS matching_tokens_frequency,
            FROM
                sequences_aggregated AS s
                LEFT JOIN single_token_frequencies AS stf
                    ON stf.text_id = s.text_id
            GROUP BY s.text_id
            ORDER BY matching_tokens_frequency DESC
            LIMIT {str(results_number)}
        '''
    ).arrow()

    return text_id_arrow_table
