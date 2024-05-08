from .base import BaseModel


class TextQuality(BaseModel):
    quality_index: float
    quality_reason: str
    ethical_index: float
    ethical_reason: str
    category: str
    language: str
