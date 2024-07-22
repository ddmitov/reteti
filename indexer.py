#!/usr/bin/env python3

from typing import List

import duckdb
from huggingface_hub import hf_hub_download

from reteti import reteti_batch_indexer

# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app reteti python /app/indexer.py

# Input data settings:
TEXTS_NUMBER           = 100000
TEXTS_PER_BATCH_NUMBER =   1000


def newlines_remover(text: str) -> str:
    return text.replace('\n', ' ')


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

    print('')
    print('Pre-processing data ...')
    print('')

    duckdb.create_function('newlines_remover', newlines_remover)

    duckdb.sql('CREATE SEQUENCE text_id_maker START 1')

    texts_list = duckdb.sql(
        f'''
            SELECT
                nextval('text_id_maker') AS text_id,
                date_publish_final AS date,
                newlines_remover(title) AS title,
                newlines_remover(maintext) AS text
            FROM read_json_auto("/app/data/hf/2021/bg.jsonl.gz")
            WHERE
                date_publish_final IS NOT NULL
                AND title IS NOT NULL
                AND maintext IS NOT NULL
                AND title NOT LIKE '%...'
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
    metadata_column_names = ['date', 'title']

    reteti_batch_indexer(text_batches_list, metadata_column_names)

    return True


if __name__ == '__main__':
    main()
