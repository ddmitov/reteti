#!/usr/bin/env python3

# Python core module:
import io
from   multiprocessing      import cpu_count
from   multiprocessing.pool import ThreadPool

# Python PIP modules:
import duckdb
from   minio                import Minio
import pyarrow              as     pa

# Reteti core module:
from reteti_core import reteti_list_splitter


def reteti_text_uploader(
    object_storage_client: Minio,
    bucket_name:           str,
    prefix:                str,
    batch_table:           pa.Table,
    message_header:        str
) -> True:
    text_id_list = batch_table.column('text_id').to_pylist()

    text_table_list = []

    for text_id in text_id_list:
        text_table = duckdb.query(
            f'''
                SELECT
                    text_id,
                    * EXCLUDE (text_id, text),
                    text
                FROM batch_table
                WHERE text_id = '{text_id}'
            '''
        ).arrow()

        text_table_list.append(text_table)

    text_id_batch_list = reteti_list_splitter(
        text_id_list,
        cpu_count()
    )

    text_table_batch_list = reteti_list_splitter(
        text_table_list,
        cpu_count()
    )

    text_uploader_worker_arguments = [
        (
            object_storage_client,
            bucket_name,
            prefix,
            text_id_batch,
            text_table_batch,
            core_number,
            message_header
        )
        for text_id_batch, text_table_batch, core_number in zip(
            text_id_batch_list,
            text_table_batch_list,
            range(cpu_count())
        )
    ]

    thread_pool = ThreadPool(cpu_count())

    result = thread_pool.starmap_async(
        reteti_text_uploader_worker,
        text_uploader_worker_arguments
    )

    result.get()

    return True


def reteti_text_uploader_worker(
    object_storage_client: Minio,
    bucket_name:           str,
    prefix:                str,
    text_id_batch:         list,
    text_table_batch:      list,
    core_number:           int,
    message_header:        str
) -> True:
    text_number = 0

    for text_id_item, text_table in zip(text_id_batch, text_table_batch):
        text_number += 1

        buffer = io.BytesIO()

        with pa.ipc.new_file(buffer, text_table.schema) as writer:
            writer.write_table(text_table)

        buffer.seek(0)

        try:
            object_storage_client.put_object(
                bucket_name  = bucket_name,
                object_name  = f'{prefix}/{text_id_item}/data.arrow',
                data         = buffer,
                length       = len(buffer.getvalue()),
                content_type = "application/octet-stream"
            )

            message = (
                f'{message_header} - ' +
                f'core {str(core_number)} ' +
                f'text {str(text_number)}/{str(len(text_id_batch))}'
            )

            print(message, flush=True)

        except Exception as exception:
            print(exception, flush=True)

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
            f'{text_prefix}/{text_id}/data.arrow'
        )

        arrow_buffer = io.BytesIO(remote_response.read())
        text_table = pa.ipc.open_file(arrow_buffer).read_all()
    except FileNotFoundError:
        pass

    return text_table
