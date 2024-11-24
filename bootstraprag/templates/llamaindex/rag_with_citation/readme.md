
### Project Structure

```
.
├── api_server.py
├── data
│ └── mlops.pdf
├── main.py
├── rag_with_citation.py
├── readme.md
└── requirements.txt

```
### How to run ?
- `pip install -r requirementx.txt`
- edit `.env` file accordingly
- `python main.py`

### want to expose as API ?
- `python api_server.py`
- URI: http://localhost:8000/api/v1/chat/completion
- method: POST
- payload
```json
{
  "query": "what are the problems of mlops"
}
