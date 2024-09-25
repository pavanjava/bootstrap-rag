from pydantic import BaseModel


class Payload(BaseModel):
    query: str
