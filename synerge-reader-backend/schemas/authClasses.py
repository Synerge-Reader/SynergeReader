from pydantic import BaseModel

class GoogleLoginRequest(BaseModel):
    token: str


class LoginRequest(BaseModel):
    email: str
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str

