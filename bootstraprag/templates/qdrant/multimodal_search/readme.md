## What is this project all about

this is a bootstrap project using bootstrap-rag cli tool. This project assume you have docker for desktop installed in your machine.

### Project scaffolding
```
.
├── __init__.py
├── __pycache__
├── .env
├── main.py
├── readme.md
├── requirements.txt
└── measure_retrieval_quality.py
```
- docker-compose.yml: if your machine does not have qdrant installed don't worry run this `docker-compose-dev.yml` in setups folder
  - `docker-compose -f docker-compose-dev.yml up -d`
- requirements.txt: this file has all the dependencies that a project need
- measure_retrieval_quality.py: the core logic for retrieval evaluation is present in this file
- main.py: this is the driver code to test.

### How to bring in your own custom logics
- open `measure_retrieval_quality.py` and modify your `_upset_and_index` and `compute_avg_precision_at_k` functions. 

or

- create a `new_search_file.py` and extend it from `measure_retrieval_quality.py` then override the base functionality in the new one.


