from typing import Optional

from pydantic import BaseModel


class SampleItem(BaseModel):
    id: int
    filename: str
    url: str  # served from /static when using the default samples folder
    label_hint: Optional[str] = None


class PredictResponse(BaseModel):
    sample_id: int
    predicted_index: int
    predicted_label: str
    model_name: str
    # bytes_read: int
