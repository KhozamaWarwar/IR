from pydantic import BaseModel

class Query(BaseModel):
    query: str
    dataset: str