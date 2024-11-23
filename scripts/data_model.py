# Input / output data defined as classes with appropriate attributes
# accept the input params/data payload
# output the score, class, latency

from pydantic import BaseModel
from pydantic import EmailStr, HttpUrl

class NLPDataInput(BaseModel):
    text: list[str]
    user_id: EmailStr

class NLPDataOutput(BaseModel):
    model_name: str
    text: list[str]
    labels: list[str]
    scores: list[float]
    prediction_time: int






