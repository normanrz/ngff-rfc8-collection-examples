import pathlib

import numpy as np
import zarr
from pydantic import BaseModel

from ngff_rfc8_collection_examples.common import (
    Axes,
    BaseAttrs,
    CoordinateSystem,
    PathRefZarr,
    Ref,
    Scale,
)
from ngff_rfc8_collection_examples.multiscale import (
    MultiscaleWithVersion,
    RootMultiscale,
)
from ngff_rfc8_collection_examples.pydantic_tools import collect_models
from ngff_rfc8_collection_examples.single_scales import (
    RootSingleScale,
    SingleScale,
    SingleScaleWithVersion,
)

np.random.seed(0)
def pseudo_uuid():
    return "".join(np.random.choice(list("abcdef0123456789"), size=8))

group_path = (
    pathlib.Path(__file__).parent / "gen_multiscales" / "distributed_multiscale.zarr"
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
    singl_scale = SingleScaleWithVersion(
        version="0.7dev0",
        id=id,
        name=f"scale {i}",
        path=None,
        attributes=BaseAttrs(
            coordinate_systems=[world_cs],
            coordinate_transformations=[
                Scale(
                    scale=[2**i, 2**i, 2**i],
                    input=Ref(ref=id),
                    output=Ref(ref=world_cs.id),
                )
            ],
        ),
    )
    single_scale_in_ms = SingleScale(
        id=singl_scale.id,
        name=singl_scale.name,
        path=PathRefZarr(path="./" + str(array_path.relative_to(group_path))),
    )
    array = zarr.create_array(
        store=array_path,
        shape=(64 // 2**i, 64 // 2**i, 64 // 2**i),
        dtype="uint8",
        overwrite=True,
    )
    root_sc = RootSingleScale(ome=singl_scale)
    root_sc.to_zarr(array)
    scales.append(single_scale_in_ms)
ms = MultiscaleWithVersion(
    id=pseudo_uuid(),
    version="0.7dev0",
    name="test multiscale",
    attributes=BaseAttrs(coordinate_systems=[]),
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
