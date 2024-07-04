#!/usr/bin/env python3

import datetime
import os
import signal
import time
import threading

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
import gradio as gr
from tokenizers import Tokenizer
import uvicorn

from reteti import reteti_searcher

# Start the application for local development at http://0.0.0.0:7860/ using:
# docker run --rm -it --user $(id -u):$(id -g) -v $PWD:/app -p 7860:7860 reteti python /app/searcher.py

# Start the containerized application:
# docker run --rm -it -p 7860:7860 reteti

# Load settings from .env file:
load_dotenv(find_dotenv())

# Global variables:
tokenizer     = None
last_activity = None


def text_searcher(search_request: str)-> object:
    # Update last activity date and time:
    global last_activity

    last_activity = time.time()

    # Tokenize the search request - use the already initialized tokenizer:
    global tokenizer

    query_tokenized = tokenizer.encode(
        sequence=search_request,
        add_special_tokens=False
    )

    # Search:
    search_info, search_result = None, None

    search_start_time = time.time()

    search_result = reteti_searcher(query_tokenized)

    search_time = time.time() - search_start_time
    search_info = {'Search Time': f'{search_time:.3f} s'}

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
        print(f'Initiating shutdown sequence at: {datetime.datetime.now()}')

        os.kill(os.getpid(), signal.SIGINT)


def main():
    # Object storage settings:
    os.environ['AWS_ENDPOINT']          = os.environ['ENDPOINT_S3']
    os.environ['AWS_ACCESS_KEY_ID']     = os.environ['ACCESS_KEY_ID']
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['SECRET_ACCESS_KEY']
    os.environ['AWS_REGION']            = 'us-east-1'
    os.environ['ALLOW_HTTP']            = 'True'

    # Initialize tokenizer:
    global tokenizer
    tokenizer = Tokenizer.from_file('/tokenizer/tokenizer.json')

    # Disable Gradio telemetry:
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

    # Define Gradio user interface:
    search_request_box=gr.Textbox(lines=1, label='Search Request')

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

        .search {font-size: 16px !important}
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
                ## Serverless Keyword Search
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
                    **Dataset:** Common Crawl News - 2021 Bulgarian  
                    https://commoncrawl.org/blog/news-dataset-available  
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
            gr.Examples(
                [
                    'Бойко Борисов',
                    'Румен Радев',
                    'околна среда',
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
            inputs=[search_request_box],
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
