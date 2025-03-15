Reteti
--------------------------------------------------------------------------------

<img align="left" width="100" height="100" src="assets/giraffe_svgrepo_com.png">
  
Reteti is a work-in-progress lexical search experiment based on partitioned index of hashed words in object storage.

## Design Objectives

* **1.** Fast lexical search with index data based entirely on object storage
* **2.** Usability in serverless or scale-to-zero applications for scalability and cost control
* **3.** Adaptability to different cloud environments or on-premise systems

## Features

- [x] Reteti splits texts to words using a pre-tokenizer from the Tokenizers Python module.

- [x] Reteti is language-agnostic and no language-specific stemmers are used.

- [x] All words are hashed and their positions are saved in a partitioned Parquet dataset under predictable file names.

- [x] Only the Parquet files of the hashes in the search request are contacted during search.  

- [x] Search is performed using DuckDB SQL.

- [x] Storage and compute are decoupled and Reteti can be used in serverless functions.

- [x] Reteti texts can be stored anywhere and the index is independent of the text storage location.

- [x] Indexing and searching are completely separate processes.

[Gradio demo](https://reteti.fly.dev/) using one million Bulgarian and English short articles is available on [Fly.io](https://fly.io/).  
It is scale-to-zero capable and its object storage is managed by [Tigris Data](https://www.tigrisdata.com/).

## Search Rules

### Search Criteria

Reteti selects the ID numbers of texts that match the following criteria:

* **1.** They have the full set of unique word hashes presented in the search request.
* **2.** They have one or more sequences of word hashes identical to the sequence of word hashes of the search request.

### Ranking Criterion: Matching Words Frequency

The matching word frequency is the number of search request word hashes found in a document divided by the number of all word hashes in the document. Short documents with high number of matching words are at the top of the results list.

## Name

Reteti was one of the [giraffe calfs orphaned during a severe drought around 2018 and saved thanks to the kindness and efforts of a local community in Kenya](https://science.sandiegozoo.org/science-blog/lekiji-fupi-and-reteti).  
  
Today we use complex data processing technologies thanks to the knowledge, persistence and efforts of many people of a large global community. Just like the small Reteti, we owe much to this community and should always be thankful to its members for their goodwill and contributions!  

## [Thanks and Credits](./CREDITS.md)

## [License](./LICENSE)

This program is licensed under the terms of the Apache License 2.0.

## Author

[Dimitar D. Mitov](https://www.linkedin.com/in/dimitar-mitov-12388982/), 2024 - 2025
