"""Database engine, session factory, and the single ORM table."""

from collections.abc import AsyncIterator
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    echo=(settings.app_env == "development"),
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


class Item(Base):
    """One registered signature: identity plus its two embeddings."""

    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), nullable=False)
    user_created_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    user_modified_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resnet50_vector: Mapped[list | None] = mapped_column(Vector(4096), nullable=True)
    vgg16_vector: Mapped[list | None] = mapped_column(Vector(4096), nullable=True)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
