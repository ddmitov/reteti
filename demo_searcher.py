#!/usr/bin/env python3

# Python core modules:
import datetime
import hashlib
import os
import signal
import threading
import time
from   typing import List

# Python PIP modules:
from   dotenv     import find_dotenv
from   dotenv     import load_dotenv
from   fastapi    import FastAPI
import pyarrow.fs as     fs
import gradio     as     gr
from   tokenizers import normalizers
from   tokenizers import pre_tokenizers
import uvicorn

# Reteti core module:
from reteti_core import reteti_hash_reader
from reteti_core import reteti_searcher

# Reteti supplementary module:
from reteti_text import reteti_text_extractor

# Global variable for scale-to-zero capability
# after a period of inactivity:
last_activity = None

# Load settings from .env file:
load_dotenv(find_dotenv())


def search_request_hasher(search_request: str) -> list:
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
        hashlib.blake2b(word_tuple[0].encode(), digest_size=64).hexdigest()
        for word_tuple in pre_tokenized_search_request
    ]

    return hash_list


def text_searcher(
    search_request: str,
    results_number: int
) -> tuple[dict, dict]:
    # Update the timestamp of the last activity:
    global last_activity
    last_activity = time.time()

    # Initialize Parquet dataset filesystem:
    dataset_filesystem = None

    if os.environ.get('FLY_APP_NAME') is not None:
        dataset_filesystem = fs.S3FileSystem(
            endpoint_override = os.environ['TIGRIS_ENDPOINT_S3'],
            access_key        = os.environ['TIGRIS_ACCESS_KEY_ID'],
            secret_key        = os.environ['TIGRIS_SECRET_ACCESS_KEY'],
            scheme            = 'https'
        )
    else:
        dataset_filesystem = fs.LocalFileSystem()

    # Set buckets:
    index_bucket = None
    texts_bucket = None

    if os.environ.get('FLY_APP_NAME') is not None:
        index_bucket = os.environ['INDEX_BUCKET']
        texts_bucket = os.environ['TEXTS_BUCKET']
    else:
        index_bucket = '/app/data/reteti'
        texts_bucket = '/app/data/reteti'

    # Hash the search request:
    request_hashing_start = time.time()

    hash_list = search_request_hasher(search_request)

    request_hashing_time = round((time.time() - request_hashing_start), 3)

    # Read hashed words index data:
    index_reading_start = time.time()

    hash_table = reteti_hash_reader(
        dataset_filesystem,
        index_bucket,
        hash_list
    )

    index_reading_time = round((time.time() - index_reading_start), 3)

    # Search:
    search_start = time.time()

    text_id_table = reteti_searcher(
        hash_table,
        hash_list,
        results_number
    )

    search_time = round((time.time() - search_start), 3)

    # Extract all matching texts:
    text_extraction_start = time.time()

    search_result_dataframe = None

    if text_id_table is not None and text_id_table.num_rows > 0:
        search_result_table = reteti_text_extractor(
            dataset_filesystem,
            texts_bucket,
            text_id_table
        )

        search_result_dataframe = search_result_table.to_pandas() 

    search_result = {}

    if search_result_dataframe is None:
        search_result['Message:'] = 'No matching texts were found.'

    # The results dataframe is converted to
    # a numbered list of dictionaries with numbers starting from 1:
    if search_result_dataframe is not None:
        search_result_index = range(1, len(search_result_dataframe) + 1)
        search_result_list = search_result_dataframe.to_dict('records')

        for index, element in zip(search_result_index, search_result_list):
            search_result[str(index)] = element

    text_extraction_time = round((time.time() - text_extraction_start), 3)

    total_time = round(
        (
            request_hashing_time +
            index_reading_time +
            search_time +
            text_extraction_time
        ),
        3
    )

    info = {}
    info['search_request_hasher() . runtime in seconds'] = request_hashing_time
    info['reteti_hash_reader() .... runtime in seconds'] = index_reading_time
    info['reteti_searcher() ....... runtime in seconds'] = search_time
    info['reteti_text_extractor() . runtime in seconds'] = text_extraction_time
    info['Reteti functions combined runtime in seconds'] = total_time

    return info, search_result


def activity_inspector():
    global last_activity

    thread = threading.Timer(
        int(os.environ['INACTIVITY_CHECK_SECONDS']),
        activity_inspector
    )

    thread.daemon = True
    thread.start()

    inactivity_maximum = int(os.environ['INACTIVITY_MAXIMUM_SECONDS'])

    if time.time() - last_activity > inactivity_maximum:
        print(f'Initiated shutdown sequence at: {datetime.datetime.now()}')

        os.kill(os.getpid(), signal.SIGINT)


def main():
    # Matplotlib writable config directory,
    # Matplotlib is a dependency of Gradio:
    os.environ['MPLCONFIGDIR'] = '/app/data/.config/matplotlib'

    # Disable Gradio telemetry:
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

    # Define Gradio user interface:
    request_box = gr.Textbox(lines=1, label='Search Request')

    results_number = gr.Dropdown(
        [10, 20, 50],
        label='Maximal Number of Search Results',
        value=10
    )

    info_box = gr.JSON(label='Search Info', show_label=True)

    results_box = gr.JSON(label='Search Results', show_label=True)

    # Dark theme by default:
    javascript_code = '''
        function refresh() {
            const url = new URL(window.location);

            if (url.searchParams.get('__theme') !== 'dark') {
                url.searchParams.set('__theme', 'dark');
                window.location.href = url.href;
            }
        }
    '''

    # CSS styling:
    css_code = '''
        a:link {
            color: white;
            text-decoration: none;
        }

        a:visited {
            color: white;
            text-decoration: none;
        }

        a:hover {
            color: white;
            text-decoration: none;
        }

        a:active {
            color: white;
            text-decoration: none;
        }

        .dark {font-size: 16px !important}
    '''

    # Initialize Gradio interface:
    gradio_interface = gr.Blocks(
        theme=gr.themes.Glass(
            font=[
                "Arial",
                "sans-serif"
            ]
        ),
        js=javascript_code,
        css=css_code,
        title='Reteti'
    )

    with gradio_interface:
        with gr.Row():
            gr.Markdown(
                '''
                # Reteti
                ## Lexical Search on Object Storage
                '''
            )

        with gr.Row():
            with gr.Column(scale=30):
                gr.Markdown(
                    '''
                    **License:** Apache License 2.0.  
                    **Repository:** https://github.com/ddmitov/reteti  
                    '''
                )

            with gr.Column(scale=40):
                gr.Markdown(
                    '''
                    **Dataset:** [Common Crawl News](https://commoncrawl.org/blog/news-dataset-available) - 2021 - 1 000 000 articles  
                    https://huggingface.co/datasets/CloverSearch/cc-news-mutlilingual  
                    '''
                )

        with gr.Row():
            request_box.render()

        with gr.Row():
            with gr.Column(scale=1):
                results_number.render()

            with gr.Column(scale=3):
                gr.Examples(
                    [
                        'COVID-19 pandemic',
                        'vaccination campaign',
                        'vaccine nationalism',
                        'remote work',
                        'virtual learning',
                        'digital economy',
                        'international trade',
                        'pharmaceutical industry',
                        'ваксина срещу COVID-19',
                        'ваксина срещу коронавирус',
                        'околна среда'
                    ],
                    fn=text_searcher,
                    inputs=request_box,
                    outputs=results_box,
                    examples_per_page=11,
                    cache_examples=False
                )

        with gr.Row():
            search_button = gr.Button('Search')

            gr.ClearButton(
                [
                    info_box,
                    request_box,
                    results_box
                ]
            )

        with gr.Row():
            info_box.render()

        with gr.Row():
            results_box.render()

        gr.on(
            triggers=[request_box.submit, search_button.click],
            fn=text_searcher,
            inputs=[request_box, results_number],
            outputs=[info_box, results_box],
        )

    gradio_interface.show_api = False
    gradio_interface.queue()

    fastapi_app = FastAPI()

    fastapi_app = gr.mount_gradio_app(
        fastapi_app,
        gradio_interface,
        path='/'
    )

    # Update last activity date and time:
    global last_activity
    last_activity = time.time()

    # Start activity inspector in a separate thread
    # to implement scale-to-zero capability, i.e.
    # when there is no user activity for a predefined amount of time
    # the application will shut down.
    activity_inspector()

    try:
        uvicorn.run(
            fastapi_app,
            host = '0.0.0.0',
            port = 7860
        )
    except (KeyboardInterrupt, SystemExit):
        print('\n')

        exit(0)


if __name__ == '__main__':
    main()
