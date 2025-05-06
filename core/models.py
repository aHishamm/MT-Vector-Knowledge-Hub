from sqlmodel import Field, Session, SQLModel, create_engine, select
from passlib.hash import bcrypt_sha256
from typing import List, Optional
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import ARRAY, FLOAT

# db for vectorized documents 
class Document(SQLModel, table=True):
    __tablename__ = "vectorized_documents"
    id: int = Field(default=None, primary_key=True)
    title: str
    content: str
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(ARRAY(FLOAT), nullable=True)
    )

# db for user information
class User(SQLModel, table=True):
    __tablename__ = "users"
    id: int = Field(default=None, primary_key=True)
    username: str
    password_hash: str = Field(alias="password")
    email: str
    def set_password(self, password: str):
        self.password_hash = bcrypt_sha256.hash(password)
    def verify_password(self, password: str) -> bool:
        return bcrypt_sha256.verify(password, self.password_hash)