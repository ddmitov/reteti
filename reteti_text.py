#!/usr/bin/env python3

# Python core modules:
from datetime import timedelta
from time     import time

# Python PIP modules:
import duckdb
import pandas          as pd
import pyarrow         as pa
import pyarrow.dataset as ds
import pyarrow.fs      as fs


def reteti_text_writer(
    batch_table: pa.Table,
    logger:      object
) -> list:
    text_file_paths = []


    def text_file_visitor(written_file):
        text_file_paths.append(written_file.path)


    # Start measuring time:
    processing_start = time()

    text_arrow_table = duckdb.sql(
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
            FROM text_arrow_table
        '''
    ).fetch_arrow_table().to_pandas()['text_id_total'].iloc[0]

    ds.write_dataset(
        text_arrow_table,
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

    message = (f'Text batch written for {processing_time_string}')

    print(message, flush=True)
    logger.info(message)

    return text_file_paths


def reteti_text_extractor(
    dataset_filesystem:  fs.S3FileSystem,
    bucket:              str,
    text_id_arrow_table: pa.Table
) -> pd.DataFrame:
    text_id_list = text_id_arrow_table.to_pandas()['text_id'].to_list()

    text_paths = []

    for text_id in text_id_list:
        text_paths.append(f'{bucket}/texts/{text_id}/part-0.arrow')

    text_arrow_table = ds.dataset(
        text_paths,
        format             = 'arrow',
        filesystem         = dataset_filesystem
    ).to_table(
        use_threads        = True,
        fragment_readahead = 16
    )

    text_dataframe = None

    if text_id_arrow_table is not None and text_arrow_table is not None:
        text_dataframe = duckdb.query(
            '''
                SELECT
                    tiat.matching_tokens_frequency,
                    tiat.single_token_frequency,
                    tiat.matching_tokens,
                    tiat.hits,
                    tat.* EXCLUDE (text),
                    tat.text
                FROM
                    text_arrow_table AS tat
                    LEFT JOIN text_id_arrow_table AS tiat
                        ON tiat.text_id = tat.text_id
                ORDER BY matching_tokens_frequency DESC
            '''
        ).fetch_arrow_table().to_pandas()

    return text_dataframe
