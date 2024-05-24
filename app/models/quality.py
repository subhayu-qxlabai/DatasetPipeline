"""
This module defines the TextQuality class, which represents the quality of text.

The TextQuality class has the following attributes:
- quality_index: float - Quality of the text
- quality_reason: str - Reason for the quality of the text
- ethical_index: float - Ethical index of the text
- ethical_reason: str - Reason for the ethical of the text
- category: str - Category name of the text
- language: str - Language of the text

The TextQuality class also has a method:
- fix_category(categories: list[str]) -> TextQuality - Fixes the category of the text based on a list of categories

Example usage:

```python
from app.models.quality import TextQuality

# Create a TextQuality object
text_quality = TextQuality(
    quality_index=0.8,
    quality_reason="The text is of high quality",
    ethical_index=0.7,
    ethical_reason="The text is ethical",
    category="Science",
    language="en"
)

# Fix the category of the text
categories = ["Science", "History", "Literature"]
fixed_text_quality = text_quality.fix_category(categories)

# Access the fixed category
fixed_category = fixed_text_quality.category

# Print the fixed category
print(fixed_category)
```

"""

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
