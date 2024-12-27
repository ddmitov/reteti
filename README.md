Reteti
--------------------------------------------------------------------------------

<img align="left" width="100" height="100" src="assets/fupi_svgrepo_com.png">
  
Reteti is a work-in-progress lexical search experiment based on LLM tokenizer and partitioned index in object storage.

## Design Objectives

* **1.** Fast lexical search with index data based entirely on object storage
* **2.** Usability in serverless or scale-to-zero applications for scalability and cost control
* **3.** Adaptability to different cloud environments or on-premise systems

## Search Rules

### Search Criteria

Reteti selects the ID numbers of texts that match the following criteria:

* **1.** They have token occurences equal or higher than the token occurences of the search request.
* **2.** They have the full set of unique tokens presented in the search request.
* **3.** They have one or more sequences of all tokens from the search request positioned next to each other without gaps having first and last token matching the first and last token of the search request.

### Ranking Criterion: Matching Tokens Frequency

The matching tokens frequency is the number of matching search request tokens found in a document divided by the number of all tokens in the same document. Short documents with high number of matching tokens are at the top of the results list.

## [Thanks and Credits](./CREDITS.md)

## [License](./LICENSE)

This program is licensed under the terms of the Apache License 2.0.

## Author

[Dimitar D. Mitov](https://www.linkedin.com/in/dimitar-mitov-12388982/), 2024
