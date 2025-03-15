#!/usr/bin/env python3

# Python core modules:
from multiprocessing      import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib              import Path

# Python PIP module:
from minio import Minio

# Reteti core module:
from reteti_core import reteti_list_splitter

def reteti_file_uploader_worker(
    client:           Minio,
    bucket_name:      str,
    prefix:           str,
    directory:        str,
    file_paths_batch: list,
    batch_number:     int
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
                f'Batch {str(batch_number)} - ' +
                f'{str(file_number)}/{str(len(file_paths_batch))} - ' +
                f'{object_name}'
            )

            print(message, flush=True)

        except Exception as exception:
            print(exception, flush=True)

    return True


def reteti_file_uploader(
    client:         Minio,
    bucket_name:    str,
    prefix:         str,
    directory:      str,
    file_extension: str
) -> True:
    file_paths = list(Path(directory).rglob(f'*.{file_extension}'))
    file_paths_batch_list = reteti_list_splitter(file_paths, 8)

    thread_pool = ThreadPool(cpu_count())

    arrow_uploader_worker_arguments = [
        (
            client,
            bucket_name,
            prefix,
            directory,
            file_paths_batch,
            batch_number
        )
        for file_paths_batch, batch_number in zip(
            file_paths_batch_list,
            range(len(file_paths_batch_list))
        )
    ]

    result = thread_pool.starmap_async(
        reteti_file_uploader_worker,
        arrow_uploader_worker_arguments
    )

    result_list = result.get()

    return True


def reteti_file_cleaner(
    client:      Minio,
    bucket_name: str,
    prefix:      str
) -> True:
    files_to_delete = client.list_objects(
        bucket_name,
        prefix    = prefix,
        recursive = True
    )

    for file_object in files_to_delete:
        client.remove_object(bucket_name, file_object.object_name)

        print(str(file_object.object_name), flush=True)

    return True
