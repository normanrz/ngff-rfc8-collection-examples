from pathlib import Path
from typing import Literal

import zarr
from pydantic import BaseModel, Field

from ngff_rfc8_collection_examples.common import (
    BaseAttrs,
    NodeModel,
)
from ngff_rfc8_collection_examples.multiscale import Multiscale
from ngff_rfc8_collection_examples.single_scales import SingleScale


class Collection(
    NodeModel[Literal["collection"], BaseAttrs, "Collection | Multiscale | SingleScale"]
):
    type: Literal["collection"] = "collection"
    attributes: BaseAttrs = Field(default_factory=BaseAttrs)
    nodes: list["Collection | Multiscale | SingleScale"] = Field(default_factory=list)


class CollectionWithVersion(Collection):
    version: Literal["0.7dev0"] = "0.7dev0"


class RootCollection(BaseModel):
    ome: CollectionWithVersion

    @classmethod
    def from_zarr(cls, group: zarr.Group) -> "RootCollection":
        return cls.model_validate(group.attrs, context=group)

    @classmethod
    def from_json(
        cls, json_data: dict, context: None | Path = None
    ) -> "RootCollection":
        return cls.model_validate(json_data, context=context)

    def to_zarr(self, zarr_array: zarr.Group):
        zarr_array.attrs.update(self.model_dump(exclude_none=True))
