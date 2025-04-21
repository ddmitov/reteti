#!/usr/bin/env python3

# Python core module:
import os

# Python PIP modules:
import duckdb
import pyarrow         as pa
import pyarrow.dataset as ds
import pyarrow.fs      as fs


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
            SELECT COUNT(text_id) AS texts_total
            FROM text_table
        '''
    ).arrow().column('texts_total')[0].as_py()

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


def reteti_text_extractor(
    texts_filesystem: fs.S3FileSystem,
    texts_bucket:     str,
    texts_prefix:     str,
    text_id_table:    pa.Table
) -> pa.Table:
    text_id_list = text_id_table.column('text_id').to_pylist()

    text_paths = [
        f'{texts_bucket}/{texts_prefix}/{text_id}/part-0.arrow'
        for text_id in text_id_list
    ]

    text_table = ds.dataset(
        text_paths,
        format     = 'arrow',
        filesystem = texts_filesystem
    ).to_table(
        use_threads = True
    )

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
