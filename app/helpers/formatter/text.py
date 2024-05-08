from .config import FormatterConfig
from .messages import MessagesFormatter


class TextFormatter(MessagesFormatter):
    def __init__(
        self,
        config: FormatterConfig = FormatterConfig(),
    ):
        self.system_key = config.system.key
        self.user_key = config.user.key
        self.assistant_key = config.assistant.key
        self._keys = [config.system.key, config.user.key, config.assistant.key]
        messages: list[list[dict[str, str]]] = []
        super().__init__(
            messages,
            config,
        )
        
    def format_text(self, system: str = "", user: str = "", assistant: str = "") -> str:
        self.messages = [[
            {self.message_role_field: self.system_key, self.message_content_field: system},
            {self.message_role_field: self.user_key, self.message_content_field: user},
            {self.message_role_field: self.assistant_key, self.message_content_field: assistant},
        ]]
        return self.format().formatted_messages[-1]
    