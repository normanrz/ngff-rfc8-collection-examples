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
    Multiscale,
)
import pathlib
from ngff_rfc8_collection_examples.common import Axes, Ref, PathRefZarr
from ngff_rfc8_collection_examples.collection import (
    RootCollection,
    Collection,
    CollectionWithVersion,
)
from ngff_rfc8_collection_examples.pydantic_tools import collect_models
from pydantic import BaseModel
import numpy as np


np.random.seed(0)
def pseudo_uuid():
    return "".join(np.random.choice(list("abcdef0123456789"), size=8))

group_path = pathlib.Path(__file__).parent / "gen_collections" / "basic_collection.zarr"

root_group = zarr.open_group(group_path, mode="a")

ms1_path = group_path / "multiscale_1"
ms1_group = root_group.create_group("multiscale_1")

ms2_path = group_path / "multiscale_2"
ms2_group = root_group.create_group("multiscale_2")

world_cs = CoordinateSystem(
    id=pseudo_uuid(),
    name="world",
    axes=[
        Axes(name="z", type="space", unit="micrometer"),
        Axes(name="y", type="space", unit="micrometer"),
        Axes(name="x", type="space", unit="micrometer"),
    ],
)

# Create multiscale 1
scales = []
for i in range(3):
    id = pseudo_uuid()
    array_path = ms1_path / f"{i}"
    print(array_path)
    assert world_cs.id is not None
    scale = SingleScale(
        id=id,
        name=f"scale {i}",
        path=PathRefZarr(path="./" + str(array_path.relative_to(ms1_path))),
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
ms1 = MultiscaleWithVersion(
    id=pseudo_uuid(),
    version="0.7dev0",
    name="test multiscale 1",
    attributes=BaseAttrs(coordinate_systems=[world_cs]),
    nodes=scales,
)
ome_ms1 = RootMultiscale(ome=ms1)
ome_ms1.to_zarr(ms1_group)

# Create multiscale 2
scales = []
for i in range(3):
    id = pseudo_uuid()
    array_path = ms2_path / f"{i}"
    assert world_cs.id is not None
    scale = SingleScale(
        id=id,
        name=f"scale {i}",
        path=PathRefZarr(path="./" + str(array_path.relative_to(ms2_path))),
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
ms2 = MultiscaleWithVersion(
    id=pseudo_uuid(),
    version="0.7dev0",
    name="test multiscale 2",
    attributes=BaseAttrs(coordinate_systems=[world_cs]),
    nodes=scales,
)
ome_ms2 = RootMultiscale(ome=ms2)
ome_ms2.to_zarr(ms2_group)

# Create collection

collection = CollectionWithVersion(
    id = pseudo_uuid(),
    version="0.7dev0",
    name="basic collection",
    nodes=[
        Multiscale(
            id=ome_ms1.ome.id,
            name=ome_ms1.ome.name,
            path=PathRefZarr(path="./multiscale_1"),
        ),
        Multiscale(
            id=ome_ms2.ome.id,
            name=ome_ms2.ome.name,
            path=PathRefZarr(path="./multiscale_2"),
        ),
    ],
)
root_collection = RootCollection(ome=collection)
root_collection.to_zarr(root_group)
