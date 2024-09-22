from fastapi import APIRouter, Depends
from models.payload import Payload
from self_correction_core import SelfCorrectingRAG


self_correcting_rag = SelfCorrectingRAG(input_dir='data', show_progress=True, no_of_retries=3)

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post(path='/query')
def fetch_response(payload: Payload):
    response = self_correcting_rag.query_with_source_query_engine(query=payload.query)
    return response
