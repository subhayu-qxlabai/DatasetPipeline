from ...models.base import BaseModel


class RoleConfig(BaseModel):
    template: str
    key: str | None = None
    

class FormatterConfig(BaseModel):
    system: RoleConfig = RoleConfig(template="<<SYS>> {system} <<SYS>>", key="system")
    user: RoleConfig = RoleConfig(template="[INST] {user} [/INST]", key="user")
    assistant: RoleConfig = RoleConfig(template="{assistant}", key="assistant")
    message_role_field: str = "role"
    message_content_field: str = "content"
    separator: str = " "
    
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
    
    def __repr__(self):
        return str(self)
