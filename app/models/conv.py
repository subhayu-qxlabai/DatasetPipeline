from pydantic import Field

from .base import BaseModel
from ..constants import MessageRole, MessageField


class ConvProps(BaseModel):
    column: str | None = Field(default=None,description="Column Name. Defaults to 'null'")
    role_key: str | None = Field(default=None,description="Role key. Defaults to 'null'")
    content_key: str | None = Field(default=None,description="Contetnt key. Defaults to 'null'")
    has_system: bool = Field(default=False,description="Check whether it has system message. Defaults to 'False'")
    roles_map: dict[str, MessageRole] = Field(default_factory=dict,description="Mapping between role and role key. Defaults to '{}'")
    
    @property
    def is_valid(self):
        return not (len(self.roles_map) == 0 or self.role_key is None or self.content_key is None)

    def standardize(self, conv: list[dict[str, str]]):
        if not self.is_valid:
            return conv
        return [
            {
                MessageField.ROLE.value: self.roles_map[m[self.role_key]].value,
                MessageField.CONTENT.value: m[self.content_key],
            } 
            for m in conv
        ]
    