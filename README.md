Reteti
--------------------------------------------------------------------------------

<img align="left" width="100" height="100" src="assets/giraffe_svgrepo_com.png">
  
Reteti is a lexical search experiment using partitioned index of hashed words in object storage.  
  

## Design Objectives

* **1.** Fast lexical search with index data based entirely on object storage
* **2.** Usability in serverless or scale-to-zero applications for scalability and cost control
* **3.** Adaptability to different cloud environments or on-premise systems

## Features

- [x] All index data is stored only in object storage.

- [x] Reteti is language-agnostic and does not use language-specific stemmers.

- [x] Storage and compute are decoupled and Reteti can be used in serverless functions.

- [x] The index and text locations are independent from one another.

## Workflow

- [x] Texts are split to words using a normalizer and a pre-tokenizer from the Tokenizers Python module.

- [x] Words are hashed and their positions are saved in Arrow files under hash prefixes in object storage.

- [x] Only the Arrow files of the hashed words in the search request are contacted during search.

- [x] Words are represented by their hashes or alias integers during search.

- [x] Search is performed using DuckDB SQL.

## Word Definition

A word is any sequence of Unicode lowercase alphanumeric characters between two whitespaces.

## Demo

[Gradio demo](https://reteti.fly.dev/) is available on [Fly.io](https://fly.io/).  
It is a scale-to-zero application and its object storage is managed by [Tigris Data](https://www.tigrisdata.com/).

## Search Criteria

Reteti selects the IDs of texts that match the following criteria:

* **1.** They have the full set of unique word hashes presented in the search request.
* **2.** They have one or more sequences of word hashes identical to the sequence of word hashes in the search request.

## Ranking Criterion

Matching words frequency is the ranking criterion. It is defined as the number of search request words found in a document divided by the number of all words in the document. Short documents having high number of matching words are at the top of the search results.

## Name

Reteti was a [giraffe calf orphaned during a severe drought around 2018 in Northern Kenya and saved thanks to the kindness and efforts of a local community](https://science.sandiegozoo.org/science-blog/lekiji-fupi-and-reteti).  
  
Today we use complex data processing technologies thanks to the knowledge, persistence and efforts of many people of a large global community. Just like the small Reteti, we owe much to our community and should always be thankful to its members for their goodwill and contributions!  

## [Thanks and Credits](./CREDITS.md)

## [License](./LICENSE)

This program is licensed under the terms of the Apache License 2.0.

## Author

[Dimitar D. Mitov](https://www.linkedin.com/in/dimitar-mitov-12388982/), 2024 - 2025
