"""稳定生成元数据向量点 ID。"""

import uuid

_NAMESPACE = uuid.UUID("1c1af30c-2340-4bf8-b0f9-2f0a96c42a6a")


def build_meta_point_id(
    object_type: str,
    object_id: str,
    text_role: str,
    text: str,
) -> uuid.UUID:
    """按业务键生成稳定 UUID，保证重复构建时 Qdrant upsert 覆盖同一 point。"""

    key = f"{object_type}:{object_id}:{text_role}:{text}"
    return uuid.uuid5(_NAMESPACE, key)
