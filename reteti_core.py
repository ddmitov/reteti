#!/usr/bin/env python3

# Python core modules:
from   datetime             import timedelta
import hashlib
from   multiprocessing      import cpu_count
from   multiprocessing.pool import ThreadPool
import os
from   time                 import time
from   typing               import List

# Python PIP modules:
import duckdb
import pyarrow         as pa
import pyarrow.fs      as fs
import pyarrow.parquet as pq
from   tokenizers      import normalizers
from   tokenizers      import pre_tokenizers


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
    batches_total:     int,
    batch_number:      int,
    text_table:        pa.Table,
    duckdb_connection: object,
    logger:            object
) -> True:
    # Start measuring time:
    processing_start = time()

    # List of dictionaries:
    words_table_list = []

    # Extract 'text_id' and 'text' Arrow table columns as lists:
    text_id_list = text_table.column('text_id').to_pylist()
    text_list    = text_table.column('text').to_pylist()

    normalizer = normalizers.Sequence(
        [
            normalizers.NFD(),          # Decompose Unicode characters
            normalizers.StripAccents(), # Remove accents after decomposition
            normalizers.Lowercase()     # Convert to lowercase
        ]
    )

    normalized_texts = [
        normalizer.normalize_str(text)
        for text in text_list
    ]

    pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(behavior='removed'),
            pre_tokenizers.Digits(individual_digits=False)
        ]
    )

    pre_tokenized_texts = [
        pre_tokenizer.pre_tokenize_str(text)
        for text in normalized_texts
    ]

    words_batch = [
        [
            word_tuple[0]
            for word_tuple in pre_tokenized_text
        ]
        for pre_tokenized_text in pre_tokenized_texts
    ]

    # Iterate all texts in a batch:
    for text_id, words_list in zip(text_id_list, words_batch):
        # Dictionary of lists:
        word_positions = {}

        for index, word in enumerate(words_list):
            if word not in word_positions:
                word_positions[word] = []

            word_positions[word].append(index)

        total_words = len(words_list)

        for word in words_list:
            word_item = {}

            word_item['word']        = str(word)
            word_item['text_id']     = str(text_id)
            word_item['positions']   = word_positions[word]
            word_item['total_words'] = int(total_words)

            words_table_list.append(word_item)

    batch_words_table = pa.Table.from_pylist(words_table_list)

    batch_hash_table = duckdb_connection.sql(
        '''
            SELECT
                CRYPTO_HASH('blake2b-512', word) AS hash,
                text_id,
                positions,
                total_words
            FROM batch_words_table
            ORDER BY hash ASC
        '''
    ).arrow()

    duckdb_connection.sql(
        'INSERT INTO hash_table SELECT * FROM batch_hash_table'
    )

    # Calculate, display and log processing time:
    processing_time        = round((time() - processing_start), 3)
    processing_time_string = str(timedelta(seconds=processing_time))

    message = (
        f'Words batch {str(batch_number)}/{str(batches_total)} ' +
        f'processed for {processing_time_string}'
    )

    print(message, flush=True)
    logger.info(message)

    return True


def reteti_dataset_producer_worker(
    index_directory:   str,
    duckdb_connection: object,
    hash_list:         list,
    batch_number:      int
) -> True:
    local_duckdb_connection = duckdb_connection.cursor()

    hash_number = 0

    for hash_item in hash_list:
        hash_number += 1

        dataset_hash_table = local_duckdb_connection.sql(
            f'''
                SELECT *
                FROM hash_table
                WHERE hash = '{hash_item}'
            '''
        ).arrow()

        os.makedirs(f'{index_directory}/hashes/{hash_item}')

        pq.write_table(
            dataset_hash_table,
            f'{index_directory}/hashes/{hash_item}/data.parquet',
            filesystem = fs.LocalFileSystem()
        )

        message = (
            f'Batch {str(batch_number)} - ' +
            f'{str(hash_number)}/{str(len(hash_list))} - {hash_item}'
        )

        print(message, flush=True)

    return True


def reteti_dataset_producer(
    index_directory:   str,
    duckdb_connection: object,
    logger:            object
) -> True:
    unique_hash_list = duckdb_connection.sql(
        '''
            SELECT hash
            FROM hash_table
            GROUP BY hash
        '''
    ).arrow().column('hash').to_pylist()

    message = f'Unique hashes: {str(len(unique_hash_list))}'
    print(message, flush=True)
    logger.info(message)

    hash_list = reteti_list_splitter(unique_hash_list, cpu_count())

    thread_pool = ThreadPool(cpu_count())

    dataset_producer_worker_arguments = [
        (
            index_directory,
            duckdb_connection,
            hash_batch,
            batch_number
        )
        for hash_batch, batch_number in zip(
            hash_list,
            range(len(hash_list))
        )
    ]

    result = thread_pool.starmap_async(
        reteti_dataset_producer_worker,
        dataset_producer_worker_arguments
    )

    result.get()

    return True


def reteti_request_hasher(search_request: str) -> list:
    normalizer = normalizers.Sequence(
        [
            normalizers.NFD(),          # Decompose Unicode characters
            normalizers.StripAccents(), # Remove accents after decomposition
            normalizers.Lowercase()     # Convert to lowercase
        ]
    )

    normalized_search_request = normalizer.normalize_str(search_request)

    pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(behavior='removed'),
            pre_tokenizers.Digits(individual_digits=False)
        ]
    )

    pre_tokenized_search_request = \
        pre_tokenizer.pre_tokenize_str(normalized_search_request)

    hash_list = [
        hashlib.blake2b(word_tuple[0].encode(), digest_size=64).hexdigest()
        for word_tuple in pre_tokenized_search_request
    ]

    return hash_list


def reteti_index_reader(
    dataset_filesystem: fs.S3FileSystem,
    bucket:             str,
    hash_list:          list
) -> None | pa.Table:
    hash_set = set(hash_list)

    hash_paths = [
        f'{bucket}/hashes/{hash_item}/data.parquet'
        for hash_item in hash_set
    ]

    hash_table = None

    try:
        hash_table = pq.ParquetDataset(
            hash_paths,
            filesystem = dataset_filesystem,
            pre_buffer = True
        ).read(use_threads = True)
    except FileNotFoundError:
        pass

    return hash_table


def reteti_searcher(
    hash_table:     pa.Table,
    hash_list:      list,
    results_number: int
) -> None | pa.Table:
    hash_set  = set(hash_list)
    alias_set = set(range(len(hash_set)))

    hash_alias_dictionary = dict(zip(hash_set, alias_set))

    alias_list = [hash_alias_dictionary.get(item) for item in hash_list]
    alias_sequence_string = '#'.join(map(str, alias_list))

    hash_alias_table = pa.Table.from_arrays(
        [
            pa.array(hash_alias_dictionary.keys()),
            pa.array(hash_alias_dictionary.values())
        ],
        names=['hash', 'alias']
    )

    results_number_string = str(results_number)

    duckdb_connection = duckdb.connect(
        config = {'allocator_background_threads': True}
    )

    search_query = f'''
        WITH
            full_hash_set AS (
                SELECT
                    text_id,
                    FIRST(total_words) AS total_words,
                FROM hash_table
                GROUP BY text_id
                HAVING COUNT(DISTINCT(hash)) = {str(len(hash_set))}
            ),

            positions_by_alias AS (
                SELECT
                    hat.alias AS alias_int,
                    ht.text_id,
                    UNNEST(ht.positions) AS position
                FROM
                    hash_table AS ht
                    INNER JOIN full_hash_set AS fhs
                        ON fhs.text_id = ht.text_id
                    LEFT JOIN hash_alias_table AS hat
                        ON hat.hash = ht.hash
            ),

            positions_by_text AS (
                SELECT
                    text_id,
                    position,
                    alias_int
                FROM positions_by_alias
                GROUP BY
                    text_id,
                    position,
                    alias_int
            ),

            distances AS (
                SELECT
                    text_id,
                    alias_int,
                    position,
                    LEAD(position) OVER (
                        PARTITION BY text_id
                        ORDER BY position ASC
                        ROWS BETWEEN CURRENT ROW and 1 FOLLOWING
                    ) - position AS lead,
                FROM positions_by_text
            ),

            borders AS (
                SELECT
                    text_id,
                    position,
                    CASE
                        WHEN lead > 1
                        THEN CONCAT(CAST(alias_int AS VARCHAR), '##')
                        ELSE CAST(alias_int AS VARCHAR)
                    END AS alias_string
                FROM distances
            ),

            texts AS (
                SELECT
                    text_id,
                    STRING_AGG(alias_string, '#' ORDER BY position) AS text
                FROM borders
                GROUP BY text_id
            ),

            sequences AS (
                SELECT
                    text_id,
                    UNNEST(STRING_SPLIT(text, '###')) AS sequence
                FROM texts
            )

        SELECT
            -- '{alias_sequence_string}' AS req_sequence,
            -- STRING_AGG(s.sequence, '-SEP-') AS txt_sequence,
            s.text_id,
            COUNT(s.sequence) AS hits,
            hits * {str(len(hash_list))} AS matching_words,
            FIRST(fhs.total_words) AS total_words,
            ROUND(
                (matching_words / FIRST(fhs.total_words)), 5
            ) AS matching_words_frequency
        FROM
            sequences AS s
            LEFT JOIN full_hash_set AS fhs
                ON fhs.text_id = s.text_id
        WHERE sequence = '{alias_sequence_string}'
        GROUP BY s.text_id
        ORDER BY matching_words_frequency DESC
        LIMIT {results_number_string}
    '''

    return duckdb_connection.sql(search_query).arrow()
