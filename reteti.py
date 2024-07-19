#!/usr/bin/env python3

import datetime
import logging
import os
import time

from deltalake import write_deltalake
from dotenv import find_dotenv, load_dotenv
import duckdb
import pandas as pd
import pyarrow         as pa
import pyarrow.dataset as ds
import pyarrow.fs      as fs
import pyarrow.parquet as pq
import pysbd
from tokenizers import Tokenizer

load_dotenv(find_dotenv())

# Object storage settings for the deltalake module:
os.environ['AWS_ENDPOINT']               = os.environ['ENDPOINT_S3']
os.environ['AWS_ACCESS_KEY_ID']          = os.environ['ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']      = os.environ['SECRET_ACCESS_KEY']
os.environ['AWS_REGION']                 = 'us-east-1'
os.environ['AWS_S3_ALLOW_UNSAFE_RENAME'] = 'true'
os.environ['ALLOW_HTTP']                 = 'True'

bucket = os.environ['BUCKET']

# Initialize sentence segmenter:
segmenter = pysbd.Segmenter(language='bg', clean=False)


def reteti_logger_starter() -> logging.Logger:
    start_datetime_string = (
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )

    logging.basicConfig(
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        filename=f'/app/data/logs/reteti_indexer_{start_datetime_string}.log',
        filemode='a'
    )

    logger = logging.getLogger()

    return logger


def reteti_sentence_splitter(text: str) -> True:
    sentences = segmenter.segment(text)

    return sentences


def reteti_batch_indexer(
    text_batch_list: list,
    metadata_column_names: list
) -> True:
    # Start logging: 
    logger = reteti_logger_starter()
    total_time = 0

    # Initialize tokenizer:
    tokenizer = Tokenizer.from_file('/tokenizer/tokenizer.json')

    try:
        batch_number = 0

        # Iterate over all batches of texts:
        for batch in text_batch_list:
            batch_indexing_start = time.time()

            batch_number += 1

            text_list = []
            metadata_list = []
            token_list = []

            # Iterate over all texts in a batch:
            for text in batch:
                metadata_table_item = {}

                metadata_table_item['text_id'] = int(text['text_id'])

                for metadata_column_name in metadata_column_names:
                    metadata_table_item[metadata_column_name] = \
                        text[metadata_column_name]

                metadata_list.append(metadata_table_item)

                sentences = reteti_sentence_splitter(str(text['text']))
                sentence_number = 0

                # Iterate over all sentences in a text:
                for sentence in sentences:
                    sentence_number += 1

                    text_table_item = {}

                    text_table_item['text_id']     = int(text['text_id'])
                    text_table_item['sentence_id'] = int(sentence_number)
                    text_table_item['sentence']    = str(sentence)

                    # Prepare text-level list of dictionaries:
                    text_list.append(text_table_item)

                    # Tokenize every sentence:
                    tokenized_text = tokenizer.encode(
                        sequence=sentence,
                        add_special_tokens=False
                    )

                    token_ids = tokenized_text.ids

                    # Calculate token frequencies and
                    # store them in a dictionary:
                    token_frequencies = {
                        token_id: token_ids.count(token_id)
                        for token_id in set(token_ids)
                    }

                    for token_id, token_frequency in token_frequencies.items():
                        token_item = {}

                        token_item['token']       = int(token_id)
                        token_item['text_id']     = int(text['text_id'])
                        token_item['sentence_id'] = int(sentence_number)
                        token_item['frequency']   = int(token_frequency)

                        token_list.append(token_item)

            # Add data to the metadata table:
            metadata_dataframe = pd.DataFrame(metadata_list)

            unique_texts = len(metadata_dataframe['text_id'].tolist())

            write_deltalake(
                f's3://{bucket}/metadata',
                metadata_dataframe,
                partition_by=['text_id'],
                max_partitions=unique_texts,
                mode='append'
            )

            # Add data to the text table:
            text_dataframe = pd.DataFrame(text_list)

            text_dataframe.sort_values(
                by=[
                    'text_id',
                    'sentence_id'
                ],
                ascending=True,
                inplace=True
            )

            unique_texts = len(text_dataframe['text_id'].unique().tolist())

            write_deltalake(
                f's3://{bucket}/texts',
                text_dataframe,
                partition_by=['text_id'],
                max_partitions=unique_texts,
                mode='append'
            )

            # Add data to the tokens table:
            tokens_dataframe = pd.DataFrame(token_list)

            tokens_dataframe.sort_values(
                by=['token'],
                ascending=True,
                inplace=True
            )

            unique_tokens = len(tokens_dataframe['token'].unique().tolist())

            write_deltalake(
                f's3://{bucket}/tokens',
                tokens_dataframe,
                partition_by=['token'],
                max_partitions=unique_tokens,
                mode='append'
            )

            # Calculate the batch processing time:
            batch_indexing_end = time.time()

            batch_indexing_time = round(
                (batch_indexing_end - batch_indexing_start),
                3
            )

            batch_indexing_time_string = str(
                datetime.timedelta(seconds=batch_indexing_time)
            )

            # Calculate the total processing time:
            total_time = round((total_time + batch_indexing_time), 3)
            total_time_string = str(datetime.timedelta(seconds=total_time))

            # Display and log the batch processing time:
            print(
                f'batch {batch_number}/{len(text_batch_list)} ' +
                f'tokenized for {batch_indexing_time_string}'
            )

            logger.info(
                f'batch {batch_number}/{len(text_batch_list)} ' +
                f'tokenized for {batch_indexing_time_string}'
            )

        # Display and log the total time just before the script exits:
        print('')
        print(f'total time: {total_time_string}')
        print('')

        logger.info(f'total time: {total_time_string}')
    except (KeyboardInterrupt, SystemExit):
        print('\n')
        exit(0)

    return True


def reteti_searcher(query_tokenized):
    # Set object storage settings:
    parquet_dataset_filesystem = fs.S3FileSystem(
        endpoint_override=os.environ['ENDPOINT_S3'],
        access_key=os.environ['ACCESS_KEY_ID'],
        secret_key=os.environ['SECRET_ACCESS_KEY'],
        scheme='http'
    )

    # Get all query tokens:
    token_list = query_tokenized.ids

    # Step 1 - read all token tables:
    token_arrow_tables_list = []

    for token in token_list:
        token_arrow_table = pq.ParquetDataset(
            f'{bucket}/tokens/token={token}/',
            filesystem=parquet_dataset_filesystem
        ).read(
            columns=[
                'text_id',
                'sentence_id',
                'frequency'
            ],
            use_threads=True
        )

        token_arrow_tables_list.append(token_arrow_table)

    tokens_arrow_table = pa.concat_tables(token_arrow_tables_list)

    # Step 2 - get the IDs of the top N matching texts:
    top_results_arrow_table = duckdb.sql(
        f'''
            SELECT
                text_id,
                sentence_id,
                SUM(frequency) AS matching_tokens
            FROM tokens_arrow_table
            GROUP BY
                text_id,
                sentence_id
            HAVING matching_tokens >= {len(token_list)}
            ORDER BY matching_tokens DESC
            LIMIT 10
        '''
    ).arrow()

    top_texts_list = duckdb.query(
        f'''
            SELECT text_id
            FROM top_results_arrow_table
        '''
    ).fetch_arrow_table().to_pandas()['text_id'].to_list()

    # Step 3 - get the top N matching texts:
    text_sql = ''
    metadata_sql = ''

    text_number = 0

    for text_id in top_texts_list:
        text_number += 1

        text_arrow_table = pq.ParquetDataset(
            f'{bucket}/texts/text_id={text_id}/',
            filesystem=parquet_dataset_filesystem
        ).read(
            columns=[
                'sentence_id',
                'sentence'
            ],
            use_threads=True
        )

        metadata_arrow_table = pq.ParquetDataset(
            f'{bucket}/metadata/text_id={text_id}/',
            filesystem=parquet_dataset_filesystem
        ).read(
            columns=[
                'date',
                'title'
            ],
            use_threads=True
        )

        # All text Arrow tables have dynamically generated variable names:
        locals()[f'text_arrow_table_{str(text_id)}']     = text_arrow_table
        locals()[f'metadata_arrow_table_{str(text_id)}'] = metadata_arrow_table

        text_sql += str(
            f'''
                SELECT
                    {text_id} AS text_id,
                    sentence_id,
                    sentence
                FROM text_arrow_table_{str(text_id)}
            '''
        )

        metadata_sql += str(
            f'''
                SELECT
                    {text_id} AS text_id,
                    date,
                    title
                FROM metadata_arrow_table_{str(text_id)}
            '''
        )

        if text_number < len(top_texts_list):
            text_sql += str(
                f'''
                    UNION
                '''
            )

            metadata_sql += str(
                f'''
                    UNION
                '''
            )

    texts_arrow_table    = duckdb.sql(text_sql).arrow()
    metadata_arrow_table = duckdb.sql(metadata_sql).arrow()

    # Step 4 - get the final search result:
    search_result_dataframe = duckdb.query(
        f'''
            SELECT
                CAST(trat.matching_tokens AS INT) AS matching_tokens,
                trat.text_id,
                mat.date,
                mat.title,
                tat.sentence_id,
                tat.sentence
            FROM
                top_results_arrow_table trat
                LEFT JOIN metadata_arrow_table mat ON
                    mat.text_id = trat.text_id
                LEFT JOIN texts_arrow_table tat ON
                    tat.text_id = trat.text_id
                    AND tat.sentence_id = trat.sentence_id
            ORDER BY matching_tokens DESC
        '''
    ).fetch_arrow_table().to_pandas()

    search_result = search_result_dataframe.to_dict('records')

    return search_result
