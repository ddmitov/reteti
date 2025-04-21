#!/usr/bin/env python3

# Python core modules:
from   datetime             import datetime
import hashlib
from   multiprocessing      import cpu_count
from   multiprocessing.pool import ThreadPool
import os
from   pathlib              import Path
from   typing               import List

# Python PIP modules:
import duckdb
from   minio import Minio
import pyarrow         as pa
import pyarrow.fs      as fs
import pyarrow.parquet as pq
from   tokenizers      import normalizers
from   tokenizers      import pre_tokenizers


def reteti_list_splitter(
    input_list:   list,
    parts_number: int
) -> List[list]:
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
    text_table:         pa.Table,
    index_root_path:    str,
    metadata_root_path: str
) -> True:
    hash_path_list = []


    def hash_file_visitor(written_file):
        hash_path_list.append(written_file.path)


    # Extract the 'text_id' and 'text' Arrow table columns as lists:
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

    # List of dictionaries:
    hashes = []

    # Iterate all texts in a batch:
    for text_id, word_list in zip(text_id_list, words_batch):
        # Hash every word in every text:
        hash_list = [
            hashlib.blake2b(word.encode(), digest_size=64).hexdigest()
            for word in word_list
        ]

        # Dictionary of lists:
        positions = {}

        for index, hashed_word in enumerate(hash_list):
            if hashed_word not in positions:
                positions[hashed_word] = []

            positions[hashed_word].append(index)

        total_words = len(hash_list)

        for hashed_word in hash_list:
            hashed_word_item = {}

            hashed_word_item['hash']        = str(hashed_word)
            hashed_word_item['text_id']     = str(text_id)
            hashed_word_item['positions']   = positions[hashed_word]
            hashed_word_item['total_words'] = int(total_words)

            hashes.append(hashed_word_item)

    hash_table = pa.Table.from_pylist(hashes)

    hash_table = duckdb.sql(
        '''
            SELECT
                hash,
                text_id,
                positions,
                total_words
            FROM hash_table
            ORDER BY hash ASC
        '''
    ).arrow()

    partitions_number = duckdb.query(
        '''
            SELECT COUNT(DISTINCT hash) AS partitions
            FROM hash_table
        '''
    ).arrow().column('partitions')[0].as_py()

    diff_string = datetime.now().strftime('%Y-%m-%d--%H-%M-%S').strip()

    os.makedirs(index_root_path, exist_ok=True)

    pq.write_to_dataset(
        hash_table,
        filesystem             = fs.LocalFileSystem(),
        root_path              = index_root_path,
        partitioning           = ['hash'],
        basename_template      = 'part-{{i}}--{}.parquet'.format(diff_string),
        existing_data_behavior = 'overwrite_or_ignore',
        max_partitions         = int(partitions_number),
        file_visitor           = hash_file_visitor
    )

    # Add data to the metadata dataset:
    metadata_table = pa.Table.from_arrays(
        [pa.array(hash_path_list)],
        names = ['path']
    )

    metadata_table = duckdb.sql(
        '''
            SELECT
                REGEXP_EXTRACT(path, '[0-9a-f]{128}')  AS hash,
                REGEXP_EXTRACT(path, 'part.*parquet$') AS filename
            FROM metadata_table
        '''
    ).arrow()

    os.makedirs(metadata_root_path, exist_ok=True)

    pq.write_table(
        metadata_table,
        f'{metadata_root_path}/metadata--{diff_string}.parquet',
        filesystem = fs.LocalFileSystem()
    )

    return True


def reteti_index_compactor(
    index_filesystem: fs.S3FileSystem,
    index_bucket:     str,
    index_prefix:     str,
    metadata_prefix:  str
) -> True:
    metadata_table = pq.ParquetDataset(
        f'{index_bucket}/{metadata_prefix}',
        filesystem = index_filesystem
    ).read()

    non_fragmented_hashes_table = duckdb.query(
        '''
            SELECT
                hash,
                COUNT(filename) AS files
            FROM metadata_table
            GROUP BY hash
            HAVING files = 1
        '''
    ).arrow()

    compaction_candidates_table = duckdb.query(
        '''
            SELECT
                hash,
                COUNT(filename) AS files
            FROM metadata_table
            GROUP BY hash
            HAVING files > 1
        '''
    ).arrow()

    compaction_candidates_list = \
        compaction_candidates_table.column('hash').to_pylist()

    if len(compaction_candidates_list) == 0:
        return True

    hash_batch_list = reteti_list_splitter(
        compaction_candidates_list,
        cpu_count()
    )

    thread_pool = ThreadPool(cpu_count())

    index_compactor_worker_arguments = [
        (
            index_filesystem,
            index_bucket,
            index_prefix,
            hash_batch,
            core_number
        )
        for hash_batch, core_number in zip(
            hash_batch_list,
            range(cpu_count())
        )
    ]

    result = thread_pool.starmap_async(
        reteti_index_compactor_worker,
        index_compactor_worker_arguments
    )

    result.get()

    updated_metadata_table = duckdb.query(
        '''
            SELECT
                mt.hash,
                mt.filename
            FROM
                metadata_table AS mt
                INNER JOIN non_fragmented_hashes_table AS nfht
                    ON nfht.hash = mt.hash
            UNION
            SELECT
                mt.hash,
                'compacted.parquet' AS filename
            FROM
                metadata_table AS mt
                INNER JOIN compaction_candidates_table AS cct
                    ON cct.hash = mt.hash
        '''
    ).arrow()

    index_filesystem.delete_dir(f'{index_bucket}/{metadata_prefix}')

    index_filesystem.create_dir(f'{index_bucket}/{metadata_prefix}')

    diff_string = datetime.now().strftime('%Y-%m-%d--%H-%M-%S').strip()

    pq.write_table(
        updated_metadata_table,
        f'{index_bucket}/{metadata_prefix}/metadata--{diff_string}.parquet',
        filesystem = index_filesystem
    )

    file_removal_table = duckdb.query(
        '''
            SELECT
                mt.hash,
                mt.filename
            FROM
                metadata_table AS mt
                INNER JOIN compaction_candidates_table AS cct
                    ON cct.hash = mt.hash
        '''
    ).arrow()

    removal_filepath_list = []

    compacted_hashes_list = file_removal_table.column('hash').to_pylist()
    files_to_remove_list  = file_removal_table.column('filename').to_pylist()

    for hash_item, filename in zip(
        compacted_hashes_list,
        files_to_remove_list
    ):
        removal_filepath_list.append(
            f'{index_bucket}/{index_prefix}/{hash_item}/{filename}'
        )

    removal_filepath_batch_list = reteti_list_splitter(
        removal_filepath_list,
        cpu_count()
    )

    file_remover_worker_arguments = [
        (
            index_filesystem,
            filepath_batch,
            core_number
        )
        for filepath_batch, core_number in zip(
            removal_filepath_batch_list,
            range(cpu_count())
        )
    ]

    result = thread_pool.starmap_async(
        reteti_file_remover_worker,
        file_remover_worker_arguments
    )

    result.get()

    return True


def reteti_index_compactor_worker(
    index_filesystem: fs.S3FileSystem,
    index_bucket:     str,
    index_prefix:     str,
    hash_list:        list,
    core_number:      int
) -> True:
    hash_number = 0

    for hash_item in hash_list:
        hash_number += 1

        hash_table = pq.ParquetDataset(
            f'{index_bucket}/{index_prefix}/{hash_item}',
            filesystem = index_filesystem
        ).read(use_threads = False)

        pq.write_table(
            hash_table,
            f'{index_bucket}/{index_prefix}/{hash_item}/compacted.parquet',
            filesystem = index_filesystem
        )

        message = (
            f'Compaction - core {str(core_number)} - ' +
            f'{str(hash_number)}/{str(len(hash_list))} - ' +
            f'{hash_item}'
        )

        print(message, flush=True)

    return True


def reteti_file_remover_worker(
    index_filesystem: fs.S3FileSystem,
    filepath_list:    list,
    core_number:      int
) -> True:
    filepath_number = 0

    for filepath in filepath_list:
        filepath_number += 1

        index_filesystem.delete_file(filepath)

        message = (
            f'Deletion - core {str(core_number)} - ' +
            f'{str(filepath_number)}/{str(len(filepath_list))}'
        )

        print(message, flush=True)

    return True


def reteti_file_uploader(
    client:         Minio,
    bucket_name:    str,
    prefix:         str,
    directory:      str,
    file_extension: str,
    message_header: str
) -> True:
    file_paths = list(Path(directory).rglob(f'*.{file_extension}'))
    file_paths_batch_list = reteti_list_splitter(file_paths, cpu_count())

    thread_pool = ThreadPool(cpu_count())

    file_uploader_worker_arguments = [
        (
            client,
            bucket_name,
            prefix,
            directory,
            file_paths_batch,
            core_number,
            message_header
        )
        for file_paths_batch, core_number in zip(
            file_paths_batch_list,
            range(cpu_count())
        )
    ]

    result = thread_pool.starmap_async(
        reteti_file_uploader_worker,
        file_uploader_worker_arguments
    )

    result_list = result.get()

    return True


def reteti_file_uploader_worker(
    client:           Minio,
    bucket_name:      str,
    prefix:           str,
    directory:        str,
    file_paths_batch: list,
    core_number:      int,
    message_header:   str
) -> True:
    file_number = 0

    for file_path in file_paths_batch:
        file_number += 1

        object_name = str(file_path).replace(f'{directory}/', '')

        try:
            client.fput_object(
                bucket_name,
                prefix + '/' + object_name,
                file_path,
                part_size = 100 * 1024 * 1024 # 100 MB
            )

            message = (
                f'{message_header} {str(core_number)} ' +
                f'{str(file_number)}/{str(len(file_paths_batch))} - ' +
                f'{object_name}'
            )

            print(message, flush=True)

        except Exception as exception:
            print(exception, flush=True)

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
    index_filesystem: fs.S3FileSystem,
    index_bucket:     str,
    index_prefix:     str,
    hash_list:        list
) -> None | pa.Table:
    hash_set  = set(hash_list)
    alias_set = set(range(len(hash_set)))

    index_reader_worker_arguments = [
        (
            index_filesystem,
            index_bucket,
            index_prefix,
            hash_item,
            hash_alias
        )
        for hash_item, hash_alias in zip(hash_set, alias_set)
    ]

    thread_pool = ThreadPool(cpu_count())

    result = thread_pool.starmap_async(
        reteti_index_reader_worker,
        index_reader_worker_arguments
    )

    result_list = result.get()

    hash_table_list = [
        result for result in result_list
        if result is not None
    ]

    hash_table = None

    if len(hash_table_list) < len(hash_set):
        return None

    if len(hash_table_list) == len(hash_set):
        hash_table = pa.concat_tables(hash_table_list)

    return hash_table


def reteti_index_reader_worker(
    index_filesystem: fs.S3FileSystem,
    index_bucket:     str,
    index_prefix:     str,
    hash_item:        str,
    hash_alias:       str
) -> None | pa.Table:
    hash_table = None

    try:
        raw_hash_table = pq.ParquetDataset(
            f'{index_bucket}/{index_prefix}/{hash_item}',
            filesystem = index_filesystem
        ).read(
            use_threads = True
        )

        alias_array = pa.array([hash_alias] * raw_hash_table.num_rows)
        hash_table = raw_hash_table.add_column(0, 'alias', alias_array)
    except FileNotFoundError:
        pass

    return hash_table


def reteti_single_word_searcher(
    hash_table:     pa.Table,
    results_number: int
) -> None | pa.Table:
    results_number_string = str(results_number)

    search_query = f'''
        SELECT
            text_id,
            LEN(FIRST(positions)) AS hits,
            hits AS matching_words,
            FIRST(total_words) AS total_words,
            ROUND(
                (matching_words / FIRST(total_words)), 5
            ) AS matching_words_frequency
        FROM hash_table
        GROUP BY text_id
        ORDER BY matching_words_frequency DESC
        LIMIT {results_number_string}
    '''

    result_table = duckdb.sql(search_query).arrow()

    if result_table.num_rows == 0:
        result_table = None

    return result_table


def reteti_multiple_words_searcher(
    hash_table:     pa.Table,
    hash_list:      list,
    results_number: int
) -> None | pa.Table:
    hash_set  = set(hash_list)
    alias_set = set(range(len(hash_set)))

    hash_alias_dictionary = dict(zip(hash_set, alias_set))
    alias_list = [hash_alias_dictionary.get(item) for item in hash_list]

    alias_sequence_string = '#'.join(map(str, alias_list))

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
                HAVING COUNT(DISTINCT(alias)) = {str(len(hash_set))}
            ),

            positions_by_alias AS (
                SELECT
                    ht.alias AS alias_int,
                    ht.text_id,
                    UNNEST(ht.positions) AS position
                FROM
                    hash_table AS ht
                    INNER JOIN full_hash_set AS fhs
                        ON fhs.text_id = ht.text_id
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
        WHERE
            sequence = '{alias_sequence_string}'
            OR sequence LIKE '%{alias_sequence_string}'
            OR sequence LIKE '{alias_sequence_string}%'
        GROUP BY s.text_id
        ORDER BY matching_words_frequency DESC
        LIMIT {results_number_string}
    '''

    result_table = duckdb_connection.sql(search_query).arrow()

    if result_table.num_rows == 0:
        result_table = None

    return result_table
