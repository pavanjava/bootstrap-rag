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
└── multimodal_search.py
```
- docker-compose.yml: if your machine does not have qdrant installed don't worry run this `docker-compose-dev.yml` in setups folder
  - `docker-compose -f docker-compose-dev.yml up -d`
- requirements.txt: this file has all the dependencies that a project need
- multimodal_search.py: the core logic for retrieval evaluation is present in this file
- main.py: this is the driver code to test.

### How to bring in your own custom logics
- open `multimodal_search.py` and modify your `search_image_by_text`, `search_text_by_image` and `_create_and_insert` functions. 

or

- create a `new_search_file.py` and extend it from `multimodal_search.py` then override the base functionality in the new one.