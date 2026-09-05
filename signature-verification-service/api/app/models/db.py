from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), nullable=False)
    user_created_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    user_modified_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resnet50_vector: Mapped[list | None] = mapped_column(Vector(4096), nullable=True)
    vgg16_vector: Mapped[list | None] = mapped_column(Vector(4096), nullable=True)
