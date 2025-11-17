import json
import uuid
from pathlib import Path
from typing import Generic, Literal, TypeVar

import urllib3.util
import zarr
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    PrivateAttr,
    model_serializer,
    model_validator,
)

from ngff_rfc8_collection_examples.pydantic_tools import collect_ids

url = urllib3.util.parse_url("https://example.com")


def random_id() -> str:
    """Generate a random UUID string."""
    return str(uuid.uuid4())


def resolve_zarr_path(
    path: str, context: zarr.Group | None = None
) -> zarr.Group | zarr.Array:
    """Resolve a path within a Zarr store."""
    print("Resolving Zarr path:", path, "with context:", context)
    if context is None:
        group = zarr.open_group(store=path, mode="a")
        return group
    elif not isinstance(context, zarr.Group):
        raise TypeError("Context must be a zarr.Group or None.")
    
    # Remove leading './' if present
    path = path.lstrip("./")
    if path.startswith("../"):
        raise ValueError("Not supported yet")
    group = context.get(path, None)
    if group is None:
        raise ValueError(f"Path '{path}' not found in the given Zarr group context.")
    return group


def resolve_local_path(path: str, context: Path | None = None) -> Path:
    """Resolve a local filesystem path."""
    if context is None:
        # Path needs to be an absolute path
        path_obj = Path(path)
        if not path_obj.is_absolute():
            raise ValueError(
                f"Path '{path}' is not absolute and cannot be resolved without context."
            )
        return path_obj
    full_path = (context.parent / path).resolve()
    if not full_path.exists():
        raise ValueError(f"Resolved path '{full_path}' does not exist.")
    return full_path


class PathRefJson(BaseModel):
    type: Literal["json"] = "json"
    path: str
    _context: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_context(self, info):
        self._context = info.context
        return self

    def resolve_path(self) -> Path:
        if isinstance(self._context, zarr.Group):
            raise TypeError("Filesystem path cannot be resolved with a Zarr context.")
        return resolve_local_path(self.path, context=self._context)


class PathRefZarr(BaseModel):
    type: Literal["zarr"] = "zarr"
    path: str
    _context: zarr.Group | Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_context(self, info):
        self._context = info.context
        return self

    def resolve_path(self) -> zarr.Group | zarr.Array:
        if isinstance(self._context, Path):
            raise TypeError("Zarr path cannot be resolved with a filesystem context.")
        return resolve_zarr_path(self.path, self._context)


PathRef = PathRefJson | PathRefZarr

NodeType = TypeVar("NodeType", bound=str)
AttrType = TypeVar("AttrType", bound=BaseModel)
NodesType = TypeVar("NodesType", bound=BaseModel | None)


class NodeModel(BaseModel, Generic[NodeType, AttrType, NodesType]):
    id: str = Field(default_factory=random_id)
    type: NodeType
    name: str | None = None
    path: PathRef | None = None
    attributes: AttrType
    nodes: list[NodesType] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def remove_empty_lists(self, handler):
        data = handler(self)
        if len(data.get("nodes", [])) == 0:
            data.pop("nodes", None)
        if len(data.get("attributes", {})) == 0:
            data.pop("attributes", None)
        return data


TargeModelType = TypeVar("TargeModelType", bound=BaseModel)


def resolve_ref_from_context(
    ref: str, context_model: BaseModel, model_type: type[TargeModelType]
) -> TargeModelType:
    """Resolve a reference string within the given context model."""
    ids_dict = collect_ids(context_model)
    if ref not in ids_dict:
        raise ValueError(f"Reference '{ref}' not found in the context model.")
    target_model = ids_dict[ref]
    if not isinstance(target_model, model_type):
        raise TypeError(
            f"Referenced model with id '{ref}' is not of type {model_type.__name__}."
        )
    return target_model


def resolve_ref_from_path(
    ref: str, path: Path, model_type: type[TargeModelType]
) -> TargeModelType:
    """Resolve a reference string from a given file path."""
    with open(path, "r") as f:
        data = json.load(f)
    model_instance = model_type.model_validate(data)
    id = getattr(model_instance, "id", None)
    if id is None:
        raise ValueError(f"Model at path '{path}' does not have an 'id' attribute.")
    if id != ref:
        raise ValueError(f"ID mismatch: expected '{ref}', found '{id}'.")
    return model_instance


class Ref(BaseModel):
    path: PathRef | None = None
    ref: str

    def resolve_ref(
        self, context_model: BaseModel, model_type: type[TargeModelType]
    ) -> TargeModelType:
        """Resolve the reference within the given context model."""
        if self.path is None:
            return resolve_ref_from_context(self.ref, context_model, model_type)
        resolved_path = self.path.resolve_path()
        if isinstance(resolved_path, Path):
            return resolve_ref_from_path(self.ref, resolved_path, model_type)
        elif isinstance(resolved_path, (zarr.Group, zarr.Array)):
            # Assuming the model is stored in a Zarr array as JSON
            raise NotImplementedError("Zarr path resolution not implemented yet.")
        else:
            raise TypeError(
                "Resolved path is neither a file path nor a Zarr group/array."
            )


class Axes(BaseModel):
    name: str
    type: Literal["space", "time", "channel"]
    unit: str | None = None


class Scale(BaseModel):
    type: Literal["scale"] = "scale"
    scale: list[float] = Field(default_factory=list)
    input: Ref
    output: Ref


class CoordinateTransformations(BaseModel):
    coordinate_transformations: list[Scale] | None = Field(default_factory=list)


class CoordinateSystem(BaseModel):
    id: str = Field(default_factory=random_id)
    name: str
    axes: list[Axes] = Field(default_factory=list)


class BaseAttrs(BaseModel):
    coordinate_systems: list[CoordinateSystem] = Field(
        default_factory=list,
        validation_alias=AliasChoices("coordinateSystems", "coordinate_systems"),
        serialization_alias="coordinateSystems",
    )
    coordinate_transformations: list[Scale] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "coordinateTransformations", "coordinate_transformations"
        ),
        serialization_alias="coordinateTransformations",
    )

    @model_serializer(mode="wrap")
    def remove_empty_lists(self, handler):
        data = handler(self)
        if len(data.get("coordinate_systems", [])) == 0:
            data.pop("coordinate_systems", None)
        if len(data.get("coordinate_transformations", [])) == 0:
            data.pop("coordinate_transformations", None)
            
        # Rename aliases back to camelCase
        if "coordinate_systems" in data:
            data["coordinateSystems"] = data.pop("coordinate_systems")
        if "coordinate_transformations" in data:
            data["coordinateTransformations"] = data.pop("coordinate_transformations")
        return data
