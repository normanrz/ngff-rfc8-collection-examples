from pydantic import BaseModel, Field
from typing import Literal
from ngff_rfc8_collection_examples.single_scales import SingleScale, RootSingleScale
from ngff_rfc8_collection_examples.common import (
    BaseAttrs,
    random_id,
    NodeModel,
)
import zarr
from pathlib import Path


class Multiscale(NodeModel[Literal["multiscale"], BaseAttrs, SingleScale]):
    id: str = Field(default_factory=random_id)
    type: Literal["multiscale"] = "multiscale"
    name: str | None = None
    attributes: BaseAttrs = Field(default_factory=BaseAttrs)
    nodes: list[SingleScale] = Field(default_factory=list)


class MultiscaleWithVersion(Multiscale):
    version: Literal["0.8"] = "0.8"


class RootMultiscale(BaseModel):
    ome: MultiscaleWithVersion

    @classmethod
    def from_zarr(cls, group: zarr.Group) -> "RootMultiscale":
        model = RootMultiscale.model_validate(group.attrs, context=group)
        # Resolve the all path references in the multiscale
        for scale in model.ome.nodes:
            if scale.path is not None:
                array = scale.path.resolve_path()
                assert isinstance(array, zarr.Array)
                scale_in_zarr = RootSingleScale.from_zarr(array)
                new_attributes = scale_in_zarr.ome.attributes.model_dump()
                new_attributes.update(scale.attributes.model_dump())
                scale.attributes = BaseAttrs.model_validate(new_attributes)
        return model

    @classmethod
    def from_json(
        cls, json_data: dict, context: None | Path = None
    ) -> "RootMultiscale":
        return cls.model_validate(json_data, context=context)

    def to_zarr(self, zarr_array: zarr.Group):
        zarr_array.attrs.update(self.model_dump(exclude_none=True))
