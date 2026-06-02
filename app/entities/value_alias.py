"""Business alias for a real enum value."""

from dataclasses import dataclass


@dataclass
class ValueAlias:
    """Maps user language to a canonical value in one catalog column."""

    column_id: str
    alias: str
    canonical_value: str
