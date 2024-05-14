from .base import BaseModel
from ..helpers.utils import find_best_match

class TextQuality(BaseModel):
    quality_index: float
    quality_reason: str
    ethical_index: float
    ethical_reason: str
    category: str
    language: str

    def fix_category(self, categories: list[str]):
        if not isinstance(categories, list) and categories:
            return self
        self.category = find_best_match(self.category, categories).text
        return self
