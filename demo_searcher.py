#!/usr/bin/env python3

# Python core modules:
import datetime
from   multiprocessing      import cpu_count
from   multiprocessing.pool import ThreadPool
import os
import signal
import time
import threading

# Python PIP modules:
from   dotenv     import find_dotenv
from   dotenv     import load_dotenv
from   fastapi    import FastAPI
import pyarrow.fs as     fs
import gradio     as     gr
from   tokenizers import Tokenizer
import uvicorn

# Reteti core module:
from reteti_core import reteti_searcher

# Reteti supplementary module:
from reteti_text import reteti_text_extractor

# Global variables:
tokenizer     = None
last_activity = None

# Create a global thread pool:
thread_pool = ThreadPool(cpu_count())

# Load settings from .env file:
load_dotenv(find_dotenv())


def dataset_filesystem_starter() -> fs.S3FileSystem:
    dataset_filesystem = None

    # Object storage settings for Fly.io deployment:
    if os.environ.get('FLY_APP_NAME') is not None:
        dataset_filesystem = fs.S3FileSystem(
            endpoint_override = os.environ['TIGRIS_ENDPOINT_S3'],
            access_key        = os.environ['TIGRIS_ACCESS_KEY_ID'],
            secret_key        = os.environ['TIGRIS_SECRET_ACCESS_KEY'],
            scheme            = 'https'
        )
    # Object storage settings for local development:
    else:
        dataset_filesystem = fs.S3FileSystem(
            endpoint_override = os.environ['LOCAL_ENDPOINT_S3'],
            access_key        = os.environ['LOCAL_ACCESS_KEY_ID'],
            secret_key        = os.environ['LOCAL_SECRET_ACCESS_KEY'],
            scheme            = 'http'
        )

    return dataset_filesystem


def text_searcher(
    search_request: str,
    results_number: int
) -> tuple[dict, dict]:
    # Update the timestamp of the last activity:
    global last_activity

    last_activity = time.time()

    # Tokenize the search request - use the already initialized tokenizer:
    global tokenizer

    # Initialize Parquet dataset filesystem in object storage:
    dataset_filesystem = dataset_filesystem_starter()

    # Object storage buckets:
    index_bucket = os.environ['INDEX_BUCKET']
    texts_bucket = os.environ['TEXTS_BUCKET']

    # Step 1 - token data extraction:
    token_search_start = time.time()

    # Search:
    text_id_arrow_table = reteti_searcher(
        dataset_filesystem,
        index_bucket,
        tokenizer,
        search_request,
        results_number,
        thread_pool
    )

    token_search_time = round((time.time() - token_search_start), 3)

    text_extraction_start = time.time()

    text_result_dataframe = None

    if text_id_arrow_table is not None:
        text_result_dataframe = reteti_text_extractor(
            dataset_filesystem,
            texts_bucket,
            text_id_arrow_table,
            thread_pool
        )

    search_result = {}

    if text_result_dataframe is None:
        search_result['Message:'] = 'No matching texts were found.'

    # The results dataframe is converted to
    # a numbered list of dictionaries with numbers starting from 1:
    if text_result_dataframe is not None:
        search_result_index = range(1, len(text_result_dataframe) + 1)
        search_result_list = text_result_dataframe.to_dict('records')

        for index, element in zip(search_result_index, search_result_list):
            search_result[str(index)] = element

    text_extraction_time = round((time.time() - text_extraction_start), 3)

    total_time = round((token_search_time + text_extraction_time), 3)

    search_info = {}
    search_info['reteti_searcher() ....... runtime in seconds'] = token_search_time
    search_info['reteti_text_extractor() . runtime in seconds'] = text_extraction_time
    search_info['Reteti functions combined runtime in seconds'] = total_time

    return search_info, search_result


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

    # Initialize the tokenizer only once when the application is started:
    global tokenizer
    tokenizer = Tokenizer.from_file('/tokenizer/tokenizer.json')

    # Disable Gradio telemetry:
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

    # Define Gradio user interface:
    search_request_box = gr.Textbox(lines = 1, label = 'Search Request')

    results_number = gr.Dropdown(
        [10, 20, 50],
        label = 'Maximal Number of Search Results',
        value = 10
    )

    search_info_box=gr.JSON(label = 'Search Info', show_label = True)

    search_results_box=gr.JSON(label = 'Search Results', show_label = True)

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
        theme = gr.themes.Glass(),
        js    = javascript_code,
        css   = css_code,
        title = 'Reteti'
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
            with gr.Column(scale = 30):
                gr.Markdown(
                    '''
                    **License:** Apache License 2.0.  
                    **Repository:** https://github.com/ddmitov/reteti  
                    '''
                )

            with gr.Column(scale = 40):
                gr.Markdown(
                    '''
                    **Dataset:** [Common Crawl News](https://commoncrawl.org/blog/news-dataset-available) - 2021 - 1 000 000 articles  
                    https://huggingface.co/datasets/CloverSearch/cc-news-mutlilingual  
                    '''
                )

            with gr.Column(scale = 30):
                gr.Markdown(
                    '''
                    **Tokenizer:** BGE-M3  
                    https://huggingface.co/Xenova/bge-m3  
                    '''
                )

        with gr.Row():
            search_request_box.render()

        with gr.Row():
            with gr.Column(scale = 1):
                results_number.render()

            with gr.Column(scale = 3):
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
                    fn                = text_searcher,
                    inputs            = search_request_box,
                    outputs           = search_results_box,
                    examples_per_page = 11,
                    cache_examples    = False
                )

        with gr.Row():
            search_button = gr.Button('Search')

            gr.ClearButton(
                [
                    search_info_box,
                    search_request_box,
                    search_results_box
                ]
            )

        with gr.Row():
            search_info_box.render()

        with gr.Row():
            search_results_box.render()

        gr.on(
            triggers = [
                search_request_box.submit,
                search_button.click
            ],
            fn       = text_searcher,
            inputs   = [
                search_request_box,
                results_number
            ],
            outputs  = [
                search_info_box,
                search_results_box
            ],
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
