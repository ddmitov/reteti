#!/usr/bin/env python3

# Python core modules:
from   datetime             import datetime
import hashlib
import io
from   multiprocessing      import cpu_count
from   multiprocessing.pool import ThreadPool
import os
from   pathlib              import Path
from   typing               import List

# Python PIP modules:
import duckdb
from   minio           import Minio
import pyarrow         as pa
import pyarrow.dataset as ds
import pyarrow.fs      as fs
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
    bins_total:           int,
    text_table:           pa.Table,
    index_base_directory: str,
    stopword_set:         set
) -> True:
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
        hash_list = [
            hashlib.blake2b(word.encode(), digest_size=64).hexdigest()
            for word in word_list
            if word not in stopword_set
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

            bin_number = (int(hashed_word, 16) % bins_total) + 1

            hashed_word_item['bin']         = int(bin_number)
            hashed_word_item['hash']        = str(hashed_word)
            hashed_word_item['text_id']     = int(text_id)
            hashed_word_item['positions']   = positions[hashed_word]
            hashed_word_item['total_words'] = int(total_words)

            hashes.append(hashed_word_item)

    hash_table = pa.Table.from_pylist(hashes)

    hash_table = duckdb.sql(
        '''
            SELECT
                CURRENT_TIMESTAMP AS generation_timestamp,
                *
            FROM hash_table
            ORDER BY bin
        '''
    ).arrow()

    diff_string = datetime.now().strftime('%Y-%m-%d--%H-%M-%S').strip()

    os.makedirs(index_base_directory, exist_ok=True)

    ds.write_dataset(
        hash_table,
        format                 = 'arrow',
        filesystem             = fs.LocalFileSystem(),
        base_dir               = index_base_directory,
        partitioning           = ['bin'],
        basename_template      = 'part-{{i}}--{}.arrow'.format(diff_string),
        existing_data_behavior = 'overwrite_or_ignore',
        max_partitions         = bins_total
    )

    return True


def reteti_index_formatter(
    bins_total:             int,
    generation_timestamp:   datetime,
    binned_index_directory: str,
    index_directory:        str
) -> True:
    bin_list = list(range(1, bins_total))

    for bin_number in bin_list:
        bin_table = ds.dataset(
            f'{binned_index_directory}/{bin_number}',
            format     = 'arrow',
            filesystem = fs.LocalFileSystem()
        ).to_table()

        hashes_table = duckdb.query(
            f'''
                SELECT hash
                FROM bin_table
                WHERE generation_timestamp > '{generation_timestamp.isoformat()}'
                GROUP BY hash
            '''
        ).arrow()

        hash_list = hashes_table.column('hash').to_pylist()

        if len(hash_list) > 0:
            hash_number = 0

            for hash_item in hash_list:
                hash_number += 1

                hash_table = duckdb.query(
                    f'''
                        SELECT
                            text_id,
                            positions,
                            total_words
                        FROM bin_table
                        WHERE hash = '{hash_item}'
                    '''
                ).arrow()

                os.makedirs(
                    f'{index_directory}/{hash_item}',
                    exist_ok=True
                )

                with pa.OSFile(
                    f'{index_directory}/{hash_item}/data.arrow',
                    'wb'
                ) as sink:
                    writer = pa.RecordBatchFileWriter(sink, hash_table.schema)
                    writer.write_table(hash_table)
                    writer.close()

                message = (
                    f'Bin {str(bin_number)}/{str(bins_total)} - ' +
                    f'{str(hash_number)}/{str(len(hash_list))} - ' +
                    f'{hash_item}'
                )

                print(message, flush=True)

    return True


def reteti_file_uploader(
    object_storage_client: Minio,
    bucket_name:           str,
    prefix:                str,
    directory:             str,
    file_extension:        str,
    message_header:        str
) -> True:
    file_paths = list(Path(directory).rglob(f'*.{file_extension}'))
    file_paths_batch_list = reteti_list_splitter(file_paths, cpu_count())

    thread_pool = ThreadPool(cpu_count())

    file_uploader_worker_arguments = [
        (
            object_storage_client,
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

    result.get()

    return True


def reteti_file_uploader_worker(
    object_storage_client: Minio,
    bucket_name:           str,
    prefix:                str,
    directory:             str,
    file_paths_batch:      list,
    core_number:           int,
    message_header:        str
) -> True:
    file_number = 0

    for file_path in file_paths_batch:
        file_number += 1

        object_name = str(file_path).replace(f'{directory}/', '')

        try:
            object_storage_client.fput_object(
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


def reteti_request_hasher(stopword_set: set, search_request: str) -> list:
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
        str(
            hashlib.blake2b(
                word_tuple[0].encode(),
                digest_size=64
            ).hexdigest()
        )
        for word_tuple in pre_tokenized_search_request
        if word_tuple[0] not in stopword_set
    ]

    return hash_list


def reteti_index_reader(
    object_storage_client: Minio,
    index_bucket:     str,
    index_prefix:     str,
    hash_list:        list
) -> None | pa.Table:
    hash_set  = set(hash_list)
    alias_set = set(range(len(hash_set)))

    index_reader_worker_arguments = [
        (
            object_storage_client,
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
    object_storage_client: Minio,
    index_bucket:     str,
    index_prefix:     str,
    hash_item:        str,
    hash_alias:       str
) -> None | pa.Table:
    hash_table = None

    try:
        remote_response = object_storage_client.get_object(
            index_bucket,
            f'{index_prefix}/{hash_item}/data.arrow'
        )

        arrow_buffer = io.BytesIO(remote_response.read())
        raw_hash_table = pa.ipc.open_file(arrow_buffer).read_all()

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
