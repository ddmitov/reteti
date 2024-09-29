#!/usr/bin/env python3

import datetime
import os
import signal
import time
import threading

from dotenv import find_dotenv
from dotenv import load_dotenv
from fastapi import FastAPI
import gradio as gr
from tokenizers import Tokenizer
import uvicorn

from reteti import reteti_searcher

# Load settings from .env file:
load_dotenv(find_dotenv())

# Global variables:
tokenizer     = None
last_activity = None


def text_searcher(
    search_request: str,
    search_type: str,
    results_number: int
) -> tuple[dict, dict]:
    # Update the timestamp of the last activity:
    global last_activity

    last_activity = time.time()

    # Tokenize the search request - use the already initialized tokenizer:
    global tokenizer

    search_request_tokenized = tokenizer.encode(
        sequence=search_request,
        add_special_tokens=False
    ).ids

    # Search:
    search_info, search_result = reteti_searcher(
        search_request_tokenized,
        search_type,
        results_number
    )

    return search_info, search_result


def activity_inspector():
    global last_activity

    thread = threading.Timer(
        int(os.environ['INACTIVITY_CHECK_SECONDS']),
        activity_inspector
    )

    thread.daemon = True
    thread.start()

    if time.time() - last_activity > int(os.environ['INACTIVITY_MAXIMUM_SECONDS']):
        print(f'Initiated shutdown sequence at: {datetime.datetime.now()}')

        os.kill(os.getpid(), signal.SIGINT)


def main():
    # Object storage settings:
    os.environ['AWS_ENDPOINT']          = os.environ['ENDPOINT_S3']
    os.environ['AWS_ACCESS_KEY_ID']     = os.environ['ACCESS_KEY_ID']
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['SECRET_ACCESS_KEY']
    os.environ['AWS_REGION']            = 'us-east-1'
    os.environ['ALLOW_HTTP']            = 'True'

    # Matplotlib writable config directory,
    # Matplotlib is a dependency of Gradio:
    os.environ['MPLCONFIGDIR'] = '/app/data/.config/matplotlib'

    # Initialize the tokenizer only once when the application is started:
    global tokenizer
    tokenizer = Tokenizer.from_file('/tokenizer/tokenizer.json')

    # Disable Gradio telemetry:
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

    # Define Gradio user interface:
    search_request_box=gr.Textbox(lines=1, label='Search Request')

    search_type = gr.Radio(
        [
            'Approximate Match',
            'Exact Match'
        ],
        value='Approximate Match',
        label='Search Type',
    )

    results_number = gr.Dropdown(
        [10, 20, 50],
        label="Number of Search Results",
        value=10
    )

    search_info_box=gr.JSON(label='Search Info', show_label=True)

    search_results_box=gr.JSON(label='Search Results', show_label=True)

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
        theme=gr.themes.Glass(),
        js=javascript_code,
        css=css_code,
        title='Reteti'
    )

    with gradio_interface:
        with gr.Row():
            gr.Markdown(
                '''
                # Reteti Demo
                ## Scale to Zero and Serverless Keyword Search
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
                    **Dataset:** [Common Crawl News](https://commoncrawl.org/blog/news-dataset-available) - 2021 Bulgarian  
                    https://huggingface.co/datasets/CloverSearch/cc-news-mutlilingual  
                    '''
                )

            with gr.Column(scale=30):
                gr.Markdown(
                    '''
                    **Tokenizer:** BGE-M3  
                    https://huggingface.co/Xenova/bge-m3  
                    '''
                )

        with gr.Row():
            search_request_box.render()

        with gr.Row():
            with gr.Column(scale=1):
                search_type.render()

            with gr.Column(scale=1):
                results_number.render()

            with gr.Column(scale=3):
                gr.Examples(
                    [
                        'ваксина срещу COVID-19',
                        'ваксина срещу коронавирус',
                        'пандемия',
                        'околна среда'
                    ],
                    fn=text_searcher,
                    inputs=search_request_box,
                    outputs=search_results_box,
                    cache_examples=False
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
            triggers=[
                search_request_box.submit,
                search_button.click
            ],
            fn=text_searcher,
            inputs=[
                search_request_box,
                search_type,
                results_number
            ],
            outputs=[
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
            host='0.0.0.0',
            port=7860
        )
    except (KeyboardInterrupt, SystemExit):
        print('\n')

        exit(0)


if __name__ == '__main__':
    main()
