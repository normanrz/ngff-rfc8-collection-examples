from pydantic import BaseModel, Field
from typing import Literal

import zarr
from ngff_rfc8_collection_examples.common import (
    CoordinateSystem,
    Scale,
    NodeModel,
    random_id,
    PathRef,
    BaseAttrs,
)


class SingleScale(NodeModel[Literal["singlescale"], BaseAttrs, None]):
    type: Literal["singlescale"] = "singlescale"
    path: PathRef | None = None
    attributes: BaseAttrs = Field(default_factory=BaseAttrs)
    nodes: list[None] = Field(
        default_factory=list, max_length=0
    )  # No child nodes allowed


class SingleScaleWithVersion(SingleScale):
    version: Literal["0.8"] = "0.8"


class RootSingleScale(BaseModel):
    ome: SingleScaleWithVersion

    @classmethod
    def from_zarr(cls, zarr_array: zarr.Array) -> "RootSingleScale":
        return cls.model_validate(zarr_array.attrs)

    def to_zarr(self, zarr_array: zarr.Array):
        if self.ome.path is not None:
            raise NotImplementedError(
                "Cannot serialize SingleScale with path reference to Zarr."
            )
        zarr_array.attrs.update(self.model_dump(exclude_none=True))


if __name__ == "__main__":
    from pathlib import Path
    from ngff_rfc8_collection_examples.common import Axes, Ref

    world_cs = CoordinateSystem(
        name="world",
        axes=[
            Axes(name="z", type="space", unit="micrometer"),
            Axes(name="y", type="space", unit="micrometer"),
            Axes(name="x", type="space", unit="micrometer"),
        ],
    )

    id = random_id()
    sc = SingleScaleWithVersion(
        id=id,
        version="0.8",
        name=f"scale {0}",
        attributes=BaseAttrs(
            coordinate_systems=[world_cs],
            coordinate_transformations=[
                Scale(
                    scale=[1, 1, 1],
                    input=Ref(ref=id),
                    output=Ref(ref=world_cs.id),
                )
            ],
        ),
    )
    ome_sc = RootSingleScale(ome=sc)

    root_sc = RootSingleScale(ome=sc)

    # Create an example in the root directory
    array_path = (
        Path(__file__).parent.parent.parent
        / "single_scale"
        / "stand_alone_single_scale.zarr"
    )

    zarr_array = zarr.create_array(
        store=array_path,
        shape=(100, 100, 100),
        dtype="uint16",
        overwrite=True,
    )
    root_sc.to_zarr(zarr_array)

    # Load it back
    loaded_sc = RootSingleScale.from_zarr(zarr_array)

    assert loaded_sc == root_sc
    from ngff_rfc8_collection_examples.pydantic_tools import collect_models

    def collect_ids_and_refs(
        ome_sc: RootSingleScale,
    ) -> tuple[dict[str, BaseModel], dict[str, Ref]]:
        models = collect_models(ome_sc.ome)
        refs: dict[str, Ref] = {}
        ids: dict[str, BaseModel] = {}
        for m in models:
            if isinstance(m, Ref):
                refs[m.ref] = m
            if hasattr(m, "id"):
                id = getattr(m, "id")
                ids[id] = m

        # Ensure that all refs point to existing ids
        for ref in refs.values():
            if ref.ref not in ids:
                raise ValueError(f"Reference {ref.ref} does not point to any known ID.")

        return ids, refs

    ids, refs = collect_ids_and_refs(ome_sc)
    print("Collected IDs:")
    for id in ids:
        print(f" - {id}")
    print("Collected Refs:")
    for ref in refs.items():
        print(f" - {ref}")
