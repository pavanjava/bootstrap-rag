from fastapi import APIRouter, Depends
from models.payload import Payload
from simple_rag import SimpleRAG

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])
simpleRAG = SimpleRAG(input_dir='data', show_progress=True)


@router.post(path='/query')
def fetch_response(payload: Payload):
    response = simpleRAG.do_rag(user_query=payload.query)
    return response
