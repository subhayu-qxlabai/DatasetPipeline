import json
from warnings import warn

from retry import retry
from datasets import Dataset
from pydantic import model_validator, Field

from .base import BaseAnalyzer, BaseConfig
from ..constants import MessageRole as Role
from ..models.quality import TextQuality
from ..models.messages import Message, Messages
from ..helpers import run_parallel_exec
from ..helpers.call_openai import call_openai_api


text_prefix = "Text to judge:\n"

TEXT_QUALITY_EXAMPLE_MESSAGES = Messages(
    messages=[
        Message(
            role=Role.SYSTEM.value,
            content="You are a helpful assistant who can judge a content and give some metrics on it.\nHere are the metrics you need to give:\n        - the quality index (0-1)\n        - the reasoning of the quality (1-2 lines)\n        - ethical index (0-1)\n        - reason for the value in ethical. (1-2 lines)\n        - the category of the content\n        - language (use ISO code: en, hi, bn, es, it, ...)\n\nReturn in JSON format\n",
        ),
        Message(
            role=Role.USER.value,
            content=f"{text_prefix}USER: My password of email account is 'abcde12345' .\nASSISTANT: okay its good but your password is not strong.",
        ),
        Message(
            role=Role.ASSISTANT.value,
            content=TextQuality(
                quality_index=0.2, 
                quality_reason='The response does not address the privacy risk of sharing passwords and lacks helpful advice on password security.', 
                ethical_index=0.0, 
                ethical_reason='The response fails to caution against sharing passwords publicly, which is a security risk.', 
                category='Digital Security', 
                language='en'
            ).to_json(indent=2),
        ),
    ]
)

class QualityConfig(BaseConfig):
    column_name: str = Field(default="messages",description="Name of the column to check the quality. Defaults to 'messages'")
    categories: list[str] | None = Field(default=None,description="List of categories to use. Defaults to 'null'")
    example_messages: Messages = Field(default=TEXT_QUALITY_EXAMPLE_MESSAGES,description=f"Defines the example messages.")
    
    @model_validator(mode="after")
    def validate_messages(self):
        assert len(self.example_messages) >= 2, "OpenAI example must have at least 2 messages"
        try:
            [
                TextQuality.from_json(x.content, fuzzy=False) 
                for x in self.example_messages.messages if x.role == Role.ASSISTANT.value
            ]
        except Exception as e:
            raise ValueError(
                f"Assistant messages for `{self.__class__.__name__}.example_messages` "
                f"must be in the following format: {TEXT_QUALITY_EXAMPLE_MESSAGES.messages[-1].content}"
            )
        return self

class QualityAnalyzer(BaseAnalyzer):
    def __init__(self, dataset: Dataset, config: QualityConfig = QualityConfig()):
        super().__init__(dataset, config)
        self.config: QualityConfig
    
    @retry(json.JSONDecodeError, tries=3, delay=3)
    def get_text_quality(self, text: str):
        response = call_openai_api(
            messages=(self.config.example_messages or TEXT_QUALITY_EXAMPLE_MESSAGES).to_list()+[
                {
                    "role": Role.USER.value,
                    "content": text_prefix + text,
                }
            ],
            temperature=0,
            n=1,
        )
        texts = [x.message.content for x in response.choices]
        tqs: list[TextQuality] = []
        for text in texts:
            try:
                tqs.append(TextQuality.from_json(text, fuzzy=True))
            except json.JSONDecodeError:
                pass
        if not tqs:
            raise ValueError(f"Failed to parse response to TextQuality: {texts}")
        return tqs[0]
    
    def _analyze(self) -> Dataset:
        if not all(isinstance(x, str) for x in self.dataset[self.config.column_name]):
            warn(f"Column {self.config.column_name!r} is not a string column. Skipping {self.name!r} analysis.")
            return self.dataset
        texts: set[str] = set(self.dataset[self.config.column_name])
        text_qualities: dict[str, TextQuality] = dict(run_parallel_exec(self.get_text_quality, texts))
        text_qualities = {k: v.fix_category(self.config.categories) for k, v in text_qualities.items()}
        return self.dataset.map(lambda x: text_qualities[x[self.config.column_name]].to_dict())
