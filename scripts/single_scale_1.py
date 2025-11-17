from pathlib import Path
from ngff_rfc8_collection_examples.common import (
    Axes,
    Ref,
    CoordinateSystem,
    Scale,
    random_id,
    BaseAttrs,
)
from ngff_rfc8_collection_examples.single_scales import (
    SingleScaleWithVersion,
    RootSingleScale,
)
import zarr
from pydantic import BaseModel
from ngff_rfc8_collection_examples.pydantic_tools import collect_models

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
root_sc = RootSingleScale(ome=sc)

# Create an example in the root directory
array_path = (
    Path(__file__).parent / "gen_single_scale" / "stand_alone_single_scale.zarr"
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


ids, refs = collect_ids_and_refs(root_sc)
print("Collected IDs:")
for id in ids:
    print(f" - {id}")
print("Collected Refs:")
for ref in refs.items():
    print(f" - {ref}")
