from typing import Any, Iterator

from pydantic import BaseModel


def iter_models(
    obj: Any, _seen: set[int] | None = None, depth: int = -1
) -> Iterator[BaseModel]:
    """Recursively iterate over all Pydantic BaseModel instances found within obj.

    The seen set is used to avoid infinite recursion on cyclic references.
    The depth parameter limits the recursion depth; -1 means unlimited.
    """
    if _seen is None:
        _seen = set()

    if depth == 0:
        return

    depth = depth - 1 if depth > 0 else depth
    if isinstance(obj, BaseModel):
        oid = id(obj)
        if oid in _seen:
            return
        _seen.add(oid)

        # yield this model first
        yield obj

        # then walk its fields
        for _, value in obj:  # iteration over model fields works in v2
            yield from iter_models(value, _seen, depth)

    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_models(v, _seen, depth)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            yield from iter_models(v, _seen, depth)


def collect_models(root: BaseModel, depth: int = -1) -> list[BaseModel]:
    return list(iter_models(root, depth=depth))


def collect_ids(root: BaseModel, depth: int = -1) -> dict[str, BaseModel]:
    """Collect all models with an 'id' attribute into a dictionary."""
    models = collect_models(root, depth=depth)
    id_dict: dict[str, BaseModel] = {}
    for m in models:
        if hasattr(m, "id"):
            id = getattr(m, "id")
            id_dict[id] = m
    return id_dict
