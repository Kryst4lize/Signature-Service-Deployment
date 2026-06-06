from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(20), default="1")
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    triton_model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_path: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), nullable=False)
    user_created_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    user_modified_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resnet50_vector: Mapped[list | None] = mapped_column(Vector(4096), nullable=True)
    vgg16_vector: Mapped[list | None] = mapped_column(Vector(4096), nullable=True)
