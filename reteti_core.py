#!/usr/bin/env python3

# Python core modules:
from   datetime             import datetime
from   datetime             import timedelta
from   multiprocessing      import cpu_count
from   multiprocessing.pool import ThreadPool
import os
import shutil
from   time                 import time
from   typing               import List

# Python PIP modules:
import duckdb
import pyarrow         as pa
import pyarrow.fs      as fs
import pyarrow.parquet as pq
from   tokenizers      import normalizers
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
    batches_total:    int,
    batch_number:     int,
    text_arrow_table: pa.Table,
    logger:           object
) -> True:
    token_paths = []


    def tokens_file_visitor(written_file):
        token_paths.append(written_file.path)


    # Start measuring time:
    processing_start = time()

    # List of dictionaries:
    token_table_list = []

    # Extract 'text_id' and 'text' Arrow table columns as lists:
    text_id_list = text_arrow_table.column('text_id').to_pylist()
    text_list    = text_arrow_table.column('text').to_pylist()

    # Tokenize all texts in the batch:
    normalizer = normalizers.Sequence(
        [
            normalizers.NFD(),          # Decompose Unicode characters
            normalizers.StripAccents(), # Remove accents after decomposition
            normalizers.Lowercase()     # Convert to lowercase
        ]
    )

    tokenizer = Tokenizer.from_file('/tokenizer/tokenizer.json')
    tokenizer.normalizer = normalizer

    token_batch = tokenizer.encode_batch_fast(
        text_list,
        add_special_tokens = False
    )

    # Combine tokens data for every text:
    for text_id, token_element in zip(text_id_list, token_batch):
        token_list = token_element.ids

        # Dictionary of lists:
        token_positions = {}

        for index, token_id in enumerate(token_list):
            if token_id not in token_positions:
                token_positions[token_id] = []

            token_positions[token_id].append(index)

        # Dictionary of integers:
        token_occurences = {
            token_id: token_list.count(token_id)
            for token_id in set(token_list)
        }

        for token_id, token_occurrences in token_occurences.items():
            total_text_tokens = len(token_list)

            token_item = {}

            token_item['token']             = int(token_id)
            token_item['text_id']           = int(text_id)
            token_item['occurrences']       = int(token_occurrences)
            token_item['positions']         = token_positions[token_id]
            token_item['total_text_tokens'] = int(total_text_tokens)

            token_table_list.append(token_item)

    # Add data to the tokens dataset:
    token_arrow_table = pa.Table.from_pylist(token_table_list)

    token_arrow_table = duckdb.sql(
        '''
            SELECT
                token AS partition,
                token,
                text_id,
                occurrences,
                positions,
                total_text_tokens
            FROM token_arrow_table
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
        root_path              = '/app/data/reteti/tokens',
        partitioning           = ['partition'],
        basename_template      = 'part-{{i}}--{}.parquet'.format(diff_string),
        existing_data_behavior = 'overwrite_or_ignore',
        max_partitions         = int(unique_partitions_number),
        file_visitor           = tokens_file_visitor
    )

    # Add data to the metadata dataset:
    path_array   = pa.array(token_paths)
    column_names = ['path']

    token_path_arrow_table = pa.Table.from_arrays(
        [path_array],
        names = column_names
    )

    token_path_arrow_table = duckdb.sql(
        '''
            SELECT
                path,
                REGEXP_EXTRACT(path, '\\d{1,10}') AS token,
                NOW() AS datetime
            FROM token_path_arrow_table
        '''
    ).arrow()

    pq.write_to_dataset(
        token_path_arrow_table,
        filesystem             = fs.LocalFileSystem(),
        root_path              = '/app/data/reteti/metadata/paths',
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
    index_directory: str,
    token_list:      list,
    batch_number:    int,
    logger:          object
) -> True:
    token_number = 0

    for token in token_list:
        token_number += 1

        token_directory = f'{index_directory}/tokens/{token}'

        token_arrow_table = pq.ParquetDataset(
            f'{token_directory}/',
            filesystem = fs.LocalFileSystem()
        ).read()

        token_files = []

        with os.scandir(token_directory) as filesystem_objects:
            for filesystem_object in filesystem_objects:
                if filesystem_object.is_file():
                    token_files.append(filesystem_object.path)

        pq.write_table(
            token_arrow_table,
            f'{token_directory}/{token}.parquet',
            filesystem  = fs.LocalFileSystem()
        )

        for token_file in token_files:
            os.remove(token_file)

        message = (
            f'Batch {str(batch_number)} - ' +
            f'{str(token_number)}/{str(len(token_list))} - ' +
            f'compacted data for token {token}'
        )

        print(message, flush=True)
        logger.info(message)

    return True


def reteti_index_compactor(
    index_directory: str,
    logger:          object
) -> True:
    metadata_directory = f'{index_directory}/metadata/paths'

    metadata_files = []

    with os.scandir(metadata_directory) as filesystem_objects:
        for filesystem_object in filesystem_objects:
            if filesystem_object.is_file():
                metadata_files.append(filesystem_object.path)

    path_arrow_table = pq.ParquetDataset(
        f'{metadata_directory}/',
        filesystem = fs.LocalFileSystem()
    ).read()

    unique_token_list = duckdb.query(
        '''
            SELECT token
            FROM path_arrow_table
            GROUP BY token
        '''
    ).fetch_arrow_table().to_pandas()['token']

    token_batch_list = reteti_list_splitter(unique_token_list, 8)

    thread_pool = ThreadPool(cpu_count())

    index_compactor_worker_arguments = [
        (
            index_directory,
            token_batch,
            batch_number,
            logger
        )
        for token_batch, batch_number in zip(
            token_batch_list,
            range(len(token_batch_list))
        )
    ]

    compaction_result = thread_pool.starmap_async(
        reteti_index_compactor_worker,
        index_compactor_worker_arguments
    )

    compaction_result.get()

    for metadata_file in metadata_files:
        os.remove(metadata_file)

    shutil.rmtree(f'{index_directory}/metadata')

    return True


def reteti_searcher(
    dataset_filesystem: fs.S3FileSystem,
    bucket:             str,
    token_list:         list,
    results_number:     int
) -> None | pa.Table:
    token_set = set(token_list)

    token_paths = []
    filters     = []

    for token in token_set:
        token_paths.append(f'{bucket}/tokens/{token}/{token}.parquet')

        occurrences = token_list.count(token)

        if occurrences > 1:
            filters.append(
                [
                    ('token', '=', token),
                    ('occurrences', '>=', token_list.count(token))
                ]
            )

    token_arrow_table = None

    if len(filters) == 0:
        try:
            token_arrow_table = pq.ParquetDataset(
                token_paths,
                filesystem = dataset_filesystem,
                pre_buffer = True
            ).read(use_threads = True)
        except FileNotFoundError as error:
            print(error, flush=True)
            pass

    if len(filters) > 0:
        try:
            token_arrow_table = pq.ParquetDataset(
                token_paths,
                filesystem = dataset_filesystem,
                filters    = filters,
                pre_buffer = True
            ).read(use_threads = True)
        except FileNotFoundError as error:
            print(error, flush=True)
            pass

    token_list_length = str(len(token_list))
    token_set_length  = str(len(token_set))

    first_token = str(token_list[0])
    last_token  = str(token_list[-1])

    distance_to_border    = str(len(token_list) - 1)
    token_sequence_string = ''.join(map(str, token_list))
    results_number_string = str(results_number)

    duckdb_connection = duckdb.connect(
        config = {'allocator_background_threads': True}
    )

    text_id_arrow_table = None

    if token_arrow_table is None:
        return text_id_arrow_table

    text_id_arrow_table = duckdb_connection.sql(
        f'''
            WITH
                full_token_set AS (
                    SELECT text_id
                    FROM token_arrow_table
                    GROUP BY text_id
                    HAVING COUNT(DISTINCT(token)) = {token_set_length}
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

                borders AS (
                    SELECT
                        text_id,
                        token,
                        position,
                        CASE
                            WHEN
                                token = {first_token}
                                AND {distance_to_border} = (
                                    LEAD(
                                        position,
                                        {distance_to_border}
                                    ) OVER (
                                        PARTITION BY text_id
                                        ORDER BY position ASC
                                    ) - position
                                )
                            THEN position
                            ELSE NULL
                        END AS start
                    FROM positions
                ),

                sequences AS (
                    SELECT
                        text_id,
                        token,
                        position,
                        MAX(start) OVER (
                            PARTITION BY text_id
                            ORDER BY position
                            ROWS BETWEEN
                                {distance_to_border} PRECEDING
                                AND CURRENT ROW
                        ) AS sequence_id
                    FROM borders
                    QUALIFY (position - sequence_id) < {token_list_length}
                ),

                sequences_aggregated AS (
                    SELECT
                        text_id,
                        sequence_id,
                        CASE
                            WHEN {token_list_length} > 2
                            THEN STRING_AGG(token, '' ORDER BY position)
                            ELSE ''
                        END AS sequence
                    FROM sequences
                    WHERE sequence_id IS NOT NULL
                    GROUP BY
                        text_id,
                        sequence_id
                    HAVING
                        COUNT(token) = {token_list_length}
                        AND FIRST(token ORDER BY position) = {first_token}
                        AND LAST(token ORDER BY position)  = {last_token}
                        AND
                        CASE
                            WHEN {token_list_length} > 2
                            THEN sequence = '{token_sequence_string}'
                            ELSE TRUE
                        END
                ),

                total_tokens AS (
                    SELECT
                        tat.text_id,
                        tat.total_text_tokens
                    FROM
                        token_arrow_table AS tat
                        INNER JOIN sequences_aggregated AS sa
                            ON sa.text_id = tat.text_id
                    GROUP BY
                        tat.text_id,
                        tat.total_text_tokens
                )

            SELECT
                sa.text_id,
                COUNT(sa.sequence_id) AS hits,
                hits * {token_list_length} AS matching_tokens,
                FIRST(tt.total_text_tokens) AS total_text_tokens,
                ROUND(
                    (matching_tokens / FIRST(tt.total_text_tokens)),
                    5
                ) AS matching_tokens_frequency,
            FROM
                sequences_aggregated AS sa
                LEFT JOIN total_tokens tt
                    ON tt.text_id = sa.text_id
            GROUP BY sa.text_id
            ORDER BY matching_tokens_frequency DESC
            LIMIT {results_number_string}
        '''
    ).arrow()

    return text_id_arrow_table
