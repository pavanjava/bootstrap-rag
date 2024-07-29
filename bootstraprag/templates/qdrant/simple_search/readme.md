## What is this project all about

this is a boot straped project using bootstrap-rag cli tool. This project assume you have docker for desktop installed in your machine.

### Project scafolding
```
.
├── __init__.py
├── __pycache__
├── docker-compose.yml
├── main.py
├── readme.md
├── requirements.txt
└── search.py
```
- docker-compose.yml: if your machine does not have qdrant installed dont worry run this docker-compose file
  - `docker-compose up -d`
- requirements.txt: this file has all the dependencies that a project need
- search.py: the core logic is present in this file
- main.py: this is the driver code to test.

### How to bring in your own custom logics
- open `search.py` and modify your `insert` and `query` functions. 

or

- create a `new_search_file.py` and extend it from `search.py` then override the base functionality in the new one.

