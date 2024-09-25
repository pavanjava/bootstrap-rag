from fastapi import APIRouter, Depends
from models.payload import Payload
from react_agent_with_query_engine import ReActWithQueryEngine


react_with_engine = ReActWithQueryEngine(input_dir='data', show_progress=True)

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post(path='/query')
def fetch_response(payload: Payload):
    response = react_with_engine.query(user_query=payload.query)
    return response
