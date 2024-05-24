from .base import BaseModel
from ..helpers.utils import find_best_match
from pydantic import Field


class TextQuality(BaseModel):
    quality_index: float=Field(description="Quality of the text")
    quality_reason: str=Field(description="Reason for the quality of the text")
    ethical_index: float=Field(description="Ethical index of the text")
    ethical_reason: str=Field(description="Reason for the ethical of the text")
    category: str=Field(description="Category name of the text")
    language: str=Field(description="Language of the text")

    def fix_category(self, categories: list[str]):
        if not isinstance(categories, list) and categories:
            return self
        self.category = find_best_match(self.category, categories).text
        return self
