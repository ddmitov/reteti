#!/usr/bin/env python3

# Python core module:
import io
from   multiprocessing      import cpu_count
from   multiprocessing.pool import ThreadPool
import os

# Python PIP modules:
import duckdb
from   minio                import Minio
import pyarrow              as     pa
import pyarrow.dataset      as     ds
import pyarrow.fs           as     fs


def reteti_text_writer(
    batch_table:    pa.Table,
    base_directory: str
) -> True:
    text_table = duckdb.sql(
        f'''
            SELECT
                text_id AS partition,
                text_id,
                * EXCLUDE (text_id, text),
                text
            FROM batch_table
        '''
    ).arrow()

    partitions_total = duckdb.query(
        '''
            SELECT COUNT(text_id) AS partitions
            FROM text_table
        '''
    ).arrow().column('partitions')[0].as_py()

    os.makedirs(base_directory, exist_ok=True)

    ds.write_dataset(
        text_table,
        format                 = 'arrow',
        filesystem             = fs.LocalFileSystem(),
        base_dir               = base_directory,
        partitioning           = ['partition'],
        basename_template      = 'part-{i}.arrow',
        existing_data_behavior = 'overwrite_or_ignore',
        max_partitions         = partitions_total
    )

    return True


def reteti_text_reader(
    object_storage_client: Minio,
    text_bucket:           str,
    text_prefix:           str,
    text_id_table:         pa.Table
) -> None | pa.Table:
    text_id_list = text_id_table.column('text_id').to_pylist()

    text_extractor_worker_arguments = [
        (
            object_storage_client,
            text_bucket,
            text_prefix,
            text_id
        )
        for text_id in text_id_list
    ]

    thread_pool = ThreadPool(cpu_count())

    result = thread_pool.starmap_async(
        reteti_text_reader_worker,
        text_extractor_worker_arguments
    )

    result_list = result.get()

    text_table_list = [
        result for result in result_list
        if result is not None
    ]

    if len(text_table_list) < len(text_id_list):
        return None

    if len(text_table_list) == len(text_id_list):
        text_table = pa.concat_tables(text_table_list)

        search_result_table = duckdb.query(
            '''
                SELECT
                    -- tit.req_sequence,
                    -- tit.txt_sequence,
                    tit.hits,
                    tit.matching_words,
                    tit.total_words,
                    tit.matching_words_frequency,
                    tt.* EXCLUDE (text),
                    tt.text
                FROM
                    text_table AS tt
                    LEFT JOIN text_id_table AS tit
                        ON tit.text_id = tt.text_id
                ORDER BY matching_words_frequency DESC
            '''
        ).arrow()

        return search_result_table


def reteti_text_reader_worker(
    object_storage_client: Minio,
    text_bucket:           str,
    text_prefix:           str,
    text_id:               str,
) -> None | pa.Table:
    try:
        remote_response = object_storage_client.get_object(
            text_bucket,
            f'{text_prefix}/{text_id}/part-0.arrow'
        )

        arrow_buffer = io.BytesIO(remote_response.read())
        text_table = pa.ipc.open_file(arrow_buffer).read_all()
    except FileNotFoundError:
        pass

    return text_table
