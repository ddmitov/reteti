#!/usr/bin/env python3

from huggingface_hub import hf_hub_download


def main():
    hf_hub_download(
        repo_id   = 'Xenova/bge-m3',
        filename  = 'tokenizer.json',
        local_dir = '/tokenizer',
        repo_type = 'model'
    )

    return True


if __name__ == '__main__':
    main()
