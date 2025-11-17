from ngff_rfc8_collection_examples.single_scales import SingleScale
from ngff_rfc8_collection_examples.common import (
    CoordinateSystem,
    Scale,
    random_id,
    BaseAttrs,
)
import zarr
from ngff_rfc8_collection_examples.multiscale import (
    MultiscaleWithVersion,
    RootMultiscale,
)
import pathlib
from ngff_rfc8_collection_examples.common import Axes, Ref, PathRef, PathRefZarr
from ngff_rfc8_collection_examples.pydantic_tools import collect_models
from pydantic import BaseModel
import numpy as np


np.random.seed(0)
def pseudo_uuid():
    return "".join(np.random.choice(list("abcdef0123456789"), size=8))

group_path = (
    pathlib.Path(__file__).parent
    / "gen_multiscales"
    / "consolidated_multiscale.zarr"
)

world_cs = CoordinateSystem(
    id=pseudo_uuid(),
    name="world",
    axes=[
        Axes(name="z", type="space", unit="micrometer"),
        Axes(name="y", type="space", unit="micrometer"),
        Axes(name="x", type="space", unit="micrometer"),
    ],
)

scales = []
for i in range(3):
    id = pseudo_uuid()
    array_path = group_path / f"{i}"
    assert world_cs.id is not None
    scale = SingleScale(
        id=id,
        name=f"scale {i}",
        path=PathRefZarr(path=str(array_path.relative_to(group_path))),
        attributes=BaseAttrs(
            coordinate_transformations=[
                Scale(
                    scale=[2**i, 2**i, 2**i],
                    input=Ref(ref=id),
                    output=Ref(ref=world_cs.id),
                )
            ],
        ),
    )
    zarr.create_array(
        store=array_path,
        shape=(64 // 2**i, 64 // 2**i, 64 // 2**i),
        dtype="uint8",
        overwrite=True,
    )
    scales.append(scale)
ms = MultiscaleWithVersion(
    id=pseudo_uuid(),
    version="0.7dev0",
    name="test multiscale",
    attributes=BaseAttrs(coordinate_systems=[world_cs]),
    nodes=scales,
)
ome_ms = RootMultiscale(ome=ms)
ome_ms.to_zarr(zarr.open_group(group_path, mode="a"))

loaded_ms = RootMultiscale.from_zarr(zarr.open_group(group_path, mode="r"))


def collect_ids_and_refs(
    ome_sc: RootMultiscale,
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


ids, refs = collect_ids_and_refs(loaded_ms)
print("Collected IDs:")
for id in ids:
    print(f" - {id}")
print("Collected Refs:")
for ref in refs.items():
    print(f" - {ref}")
