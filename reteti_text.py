#!/usr/bin/env python3

# Python core modules:
from datetime import timedelta
from time     import time

# Python PIP modules:
import duckdb
import pyarrow         as pa
import pyarrow.dataset as ds
import pyarrow.fs      as fs


def reteti_text_writer(
    batch_number:  int,
    batches_total: int,
    batch_table:   pa.Table,
    logger:        object
) -> list:
    text_file_paths = []


    def text_file_visitor(written_file):
        text_file_paths.append(written_file.path)


    # Start measuring time:
    processing_start = time()

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

    text_id_total = duckdb.query(
        '''
            SELECT COUNT(text_id) AS text_id_total
            FROM text_table
        '''
    ).arrow().column('text_id_total')[0].as_py()

    ds.write_dataset(
        text_table,
        format                 = 'arrow',
        filesystem             = fs.LocalFileSystem(),
        base_dir               = f'/app/data/reteti-texts/texts',
        partitioning           = ['partition'],
        basename_template      = 'part-{i}.arrow',
        existing_data_behavior = 'overwrite_or_ignore',
        max_partitions         = text_id_total,
        file_visitor           = text_file_visitor
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

    return text_file_paths


def reteti_text_extractor(
    dataset_filesystem: fs.S3FileSystem,
    bucket:             str,
    text_id_table:      pa.Table
) -> pa.Table:
    text_id_list = text_id_table.column('text_id').to_pylist()

    text_paths = [
        f'{bucket}/texts/{text_id}/part-0.arrow'
        for text_id in text_id_list
    ]

    text_table = ds.dataset(
        text_paths,
        format     = 'arrow',
        filesystem = dataset_filesystem
    ).to_table(
        use_threads        = True,
        fragment_readahead = 16
    )

    search_result_table = duckdb.query(
        '''
            SELECT
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
