from datasets import Dataset

from .config import FormatterConfig
from .messages import MessagesFormatter

class DatasetFormatter(MessagesFormatter):
    def __init__(
        self,
        dataset: Dataset,
        messages_column: str = "messages",
        config: FormatterConfig = FormatterConfig(),
    ):
        assert messages_column in dataset.column_names, f"{messages_column!r} not found in dataset"
        super().__init__(
            dataset[messages_column],
            config=config,
        )