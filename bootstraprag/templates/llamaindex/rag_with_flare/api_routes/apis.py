from fastapi import APIRouter, Depends
from models.payload import Payload
from base_rag import BaseRAG


base_rag = BaseRAG(show_progress=True, data_path='data')

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post(path='/query')
def fetch_response(payload: Payload):
    response = base_rag.query(query_string=payload.query)
    return response
