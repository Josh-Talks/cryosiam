from pydantic import BaseModel


class _StrictModel(BaseModel):
    class Config:
        extra = "forbid"
