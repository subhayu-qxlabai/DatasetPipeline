from pydantic import Field
from ...models.base import BaseModel


class RoleConfig(BaseModel):
    template: str = Field(description="Template to apply to the role. Example: `Some text here {value_of_key} Some text here`")
    key: str = Field(description="The key of the role. Example: `value_of_key`")
    

class FormatterConfig(BaseModel):
    system: RoleConfig = RoleConfig(template="<<SYS>> {system} <<SYS>>", key="system")
    user: RoleConfig = RoleConfig(template="[INST] {user} [/INST]", key="user")
    assistant: RoleConfig = RoleConfig(template="{assistant}", key="assistant")
    message_role_field: str = Field(default="role", description="The field name of the individual role. Defaults to 'role'")
    message_content_field: str = Field(default="content", description="The field name of the conversation text for the role. Defaults to 'content'")
    separator: str = Field(default=" ", description="The seperator to seperate the conversation texts.")
    
    def __str__(self):
        return (
            self.system.template 
            + self.separator 
            + self.user.template 
            + self.separator 
            + self.assistant.template
            + self.separator 
            + f"{self.user.template[:4]}..."
        )
