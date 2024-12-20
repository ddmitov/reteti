#!/usr/bin/env python3

from typing import List

import duckdb
from huggingface_hub import hf_hub_download

from reteti import reteti_batch_indexer

# Input data settings:
TEXTS_NUMBER           = 500000
TEXTS_PER_BATCH_NUMBER =  10000


def batch_generator(item_list: list, items_per_batch: int):
    for item in range(0, len(item_list), items_per_batch):
        yield item_list[item:item + items_per_batch]


def data_preprocessor(texts_per_batch_number: int) -> List[list]:
    # Download data from a Hugging Face dataset or open a locally cached copy:
    hf_hub_download(
        repo_id='CloverSearch/cc-news-mutlilingual',
        filename='2021/bg.jsonl.gz',
        local_dir='/app/data/hf',
        repo_type='dataset'
    )

    duckdb.sql('CREATE SEQUENCE text_id_maker START 1')

    texts_list = duckdb.sql(
        f'''
            SELECT
                nextval('text_id_maker') AS text_id,
                date_publish_final AS date,
                REPLACE(title, '\n', '') AS title,
                REPLACE(maintext, '\n', '') AS text,
            FROM read_json_auto("/app/data/hf/2021/bg.jsonl.gz")
            WHERE
                date_publish_final IS NOT NULL
                AND title IS NOT NULL
                AND title NOT LIKE '%...'
                AND LENGTH(maintext) <= 2000
            LIMIT {str(TEXTS_NUMBER)}
        '''
    ).to_arrow_table().to_pylist()

    text_batches_list = list(
        batch_generator(
            texts_list,
            texts_per_batch_number
        )
    )

    return text_batches_list


def main():
    # Pre-process input texts and group them in batches:
    text_batches_list = data_preprocessor(TEXTS_PER_BATCH_NUMBER)

    # Index all text batches:
    metadata_column_names = ['title', 'date']

    reteti_batch_indexer(text_batches_list, metadata_column_names)

    return True


if __name__ == '__main__':
    main()
