Reteti
--------------------------------------------------------------------------------

<img align="left" width="100" height="100" src="assets/giraffe_svgrepo_com.png">
  
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

## Name

Reteti was one of three [giraffe calfs orphaned during a severe drought around 2018 and saved thanks to the kindness and efforts of a local community in Kenya](https://science.sandiegozoo.org/science-blog/lekiji-fupi-and-reteti).  
  
Today we use complex data processing technologies thanks to the knowledge, persistence and efforts of many people of a large global community. Just like the small Reteti, we owe much to this community and should always be thankful to its members for their goodwill and contributions!  

## [Thanks and Credits](./CREDITS.md)

## [License](./LICENSE)

This program is licensed under the terms of the Apache License 2.0.

## Author

[Dimitar D. Mitov](https://www.linkedin.com/in/dimitar-mitov-12388982/), 2024 - 2025
