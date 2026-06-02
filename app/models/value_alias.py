"""ORM model for enum value aliases in the metadata catalog."""

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ValueAliasMySQL(Base):
    """Value alias persisted with the active metadata catalog."""

    __tablename__ = "value_alias"

    column_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    alias: Mapped[str] = mapped_column(String(128), primary_key=True)
    canonical_value: Mapped[str] = mapped_column(String(128), nullable=False)
