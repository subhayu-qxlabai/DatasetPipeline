from typing import Any

from .config import FormatterConfig


class MessagesFormatter(object):
    def __init__(
        self,
        messages: list[list[dict[str, str]]],
        config: FormatterConfig = FormatterConfig(),
        strict_validation: bool = False,
    ):
        if not isinstance(config, FormatterConfig):
            raise ValueError(f"`config` must be of type {FormatterConfig.__name__!r} but got {type(config)!r}")
        if strict_validation:
            assert f"{{{config.system.key}}}" in config.system.template, f"{{{config.system.key}}} not found in system_template"
            assert f"{{{config.user.key}}}" in config.user.template, f"{{{config.user.key}}} not found in user_template"
            assert f"{{{config.assistant.key}}}" in config.assistant.template, f"{{{config.assistant.key}}} not found in assistant_template"
        self.formatted_messages: list[str] = []
        self.tokenized_messages: list[dict[str, int | str]] = []
        
        self.system_template = config.system.template
        self.user_template = config.user.template
        self.assistant_template = config.assistant.template
        self.separator = config.separator
        self.base_format = self.separator.join([self.system_template, self.user_template, self.assistant_template])

        self.system_key = config.system.key
        self.user_key = config.user.key
        self.assistant_key = config.assistant.key

        self.messages = list(messages)
        self.message_role_field = config.message_role_field
        self.message_content_field = config.message_content_field
        # self._has_system = self.has_system

        self._formatter_map = {
            self.system_key: lambda message: self.replace_text(self.system_template, {self.system_key: message}),
            self.user_key: lambda message: self.replace_text(self.user_template, {self.user_key: message}),
            self.assistant_key: lambda message: self.replace_text(self.assistant_template, {self.assistant_key: message}),
        }

    def _apply_format(self, message_dict: dict[str, str]):
        return self._formatter_map[message_dict[self.message_role_field]](message_dict[self.message_content_field])
    
    @staticmethod
    def replace_text(text: str, replacements: dict[str, Any], curly_braces: bool = True):
        for key, value in replacements.items():
            if curly_braces:
                key = f"{{{key}}}"
            text = text.replace(key, str(value))
        return text

    @property
    def has_system(self):
        initial_roles = set(map(lambda x: x[0][self.message_role_field], self.messages))
        assert len(initial_roles) == 1, f"multiple initial roles: {initial_roles}"
        return initial_roles.pop() == self.system_key

    def format(self):
        self.formatted_messages = []
        for messages in self.messages:
            self.formatted_messages.append(
                self.separator.join(list(map(self._apply_format, messages)))
            )
        return self
    
    def tokenize(self, tokenizer, **kwargs):
        if not self.formatted_messages:
            self.format()
        self.tokenized_messages = list(map(lambda x: tokenizer(x, **kwargs), self.formatted_messages))
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(messages={len(self.messages)}, formatted={len(self.formatted_messages)}, tokenized={len(self.tokenized_messages)}, format={self.base_format!r})"
    