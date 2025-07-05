#!/usr/bin/env python3

# Python core modules:
import json
import os
import signal
import threading
import time

# Python PIP modules:
from   dotenv       import find_dotenv
from   dotenv       import load_dotenv
from   fastapi      import FastAPI
from   minio        import Minio
import gradio       as     gr
import stopwordsiso as stopwords
import uvicorn

# Reteti core module:
from reteti_core import reteti_request_hasher
from reteti_core import reteti_index_reader
from reteti_core import reteti_single_word_searcher
from reteti_core import reteti_multiple_words_searcher

# Reteti supplementary module:
from reteti_text import reteti_text_reader

# Start the application for local development at http://0.0.0.0:7860/ using:
# docker run --rm -it -p 7860:7860 \
# --user $(id -u):$(id -g) -v $PWD:/app \
# reteti-demo python /app/demo_searcher.py

# Global variable for scale-to-zero capability after a period of inactivity:
last_activity = None

# Global variable for stopwords:
stopword_set = None

# Load settings from .env file:
load_dotenv(find_dotenv())


def text_searcher(
    search_request: str,
    results_number: int
) -> tuple[dict, dict]:
    # Update the timestamp of the last activity:
    global last_activity
    last_activity = time.time()

    # Set buckets and prefixes:
    index_bucket = os.environ['INDEX_BUCKET']
    texts_bucket = os.environ['TEXTS_BUCKET']

    index_prefix = 'hashes'
    texts_prefix = 'texts'

    # Use the global stopwords set:
    global stopword_set

    # Hash the search request:
    request_hashing_start = time.time()

    hash_list = reteti_request_hasher(stopword_set, search_request)

    request_hashing_time = round((time.time() - request_hashing_start), 3)

    # Read the hashed words index data:
    index_reading_start = time.time()

    object_storage_client = Minio(
        os.environ['ENDPOINT_S3'],
        access_key = os.environ['ACCESS_KEY_ID'],
        secret_key = os.environ['SECRET_ACCESS_KEY'],
        secure     = True
    )

    hash_table = reteti_index_reader(
        object_storage_client,
        index_bucket,
        index_prefix,
        hash_list
    )

    index_reading_time = round((time.time() - index_reading_start), 3)

    # Search:
    search_start = time.time()

    text_id_table = None

    if hash_table is not None:
        if len(hash_list) == 1:
            text_id_table = reteti_single_word_searcher(
                hash_table,
                results_number
            )

        if len(hash_list) > 1:
            text_id_table = reteti_multiple_words_searcher(
                hash_table,
                hash_list,
                results_number
            )

    search_time = round((time.time() - search_start), 3)

    # Extract all matching texts:
    text_extraction_start = time.time()

    search_result_dataframe = None

    if text_id_table is not None:
        search_result_table = reteti_text_reader(
            object_storage_client,
            texts_bucket,
            texts_prefix,
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
            index_reading_time   +
            search_time          +
            text_extraction_time
        ),
        3
    )

    info = {}

    info['reteti_request_hasher() ........ runtime in seconds'] = request_hashing_time
    info['reteti_index_reader() .......... runtime in seconds'] = index_reading_time

    if len(hash_list) == 1:
        info['reteti_single_word_searcher() .. runtime in seconds'] = search_time

    if len(hash_list) > 1:
        info['reteti_multiple_words_searcher() runtime in seconds'] = search_time

    info['reteti_text_reader() ........... runtime in seconds'] = text_extraction_time
    info['Reteti functions total ......... runtime in seconds'] = total_time

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
        os.kill(os.getpid(), signal.SIGINT)


def main():
    # Get the number of indexed texts:
    object_storage_client = Minio(
        os.environ['ENDPOINT_S3'],
        access_key = os.environ['ACCESS_KEY_ID'],
        secret_key = os.environ['SECRET_ACCESS_KEY'],
        secure     = True
    )

    json_data = None

    texts_total = 0

    try:
        response = object_storage_client.get_object(
            os.environ['INDEX_BUCKET'],
            'metadata/metadata.json'
        )

        json_data = json.loads(response.read().decode('utf-8'))
    except Exception:
        pass

    try:
        texts_total = json_data['texts_total']
    except Exception:
        pass

    # Matplotlib writable config directory,
    # Matplotlib is a dependency of Gradio:
    os.environ['MPLCONFIGDIR'] = '/app/data/.config/matplotlib'

    # Disable Gradio telemetry:
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

    # Initialize a stopwords list:
    global stopword_set
    stopword_set = stopwords.stopwords(['bg', 'en'])

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
            font=[gr.themes.GoogleFont('Open Sans')],
            font_mono=[gr.themes.GoogleFont('Roboto Mono')]
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
                    **Repository:** https://github.com/ddmitov/reteti  
                    **License:** Apache License 2.0.  
                    '''
                )

            with gr.Column(scale=40):
                gr.Markdown(
                    '''
                    **Dataset:** Common Crawl News  
                    https://huggingface.co/datasets/stanford-oval/ccnews  
                    '''
                )

            with gr.Column(scale=30):
                gr.Markdown(
                    f'''
                    **Total texts:** {texts_total}  
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
                        'virtual learning',
                        'international trade',
                        'international relations',
                        'София',
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
    gradio_interface.ssr_mode = False
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
