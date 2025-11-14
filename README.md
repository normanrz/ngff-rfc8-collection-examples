# ngff-rfc8-collection-examples

This repository contains example files for the [NGFF RFC 8: Collections](https://github.com/ome/ngff/pull/343)

## Example descriptions

- `plain/base_multiscale_collection.json`: A single collection node containing two multiscale images, one of which references the other as its source.
- `plain/nested_collections.json`: A simple nested collection structure with two levels of collections. The top-level collection contains a child collection, and a multiscale image. The child collection contains another multiscale image.
- `plain/external_collection/`: This is an example of a collection that references external collections. It contains three files:
  - `parent_collection.json`: The top-level collection that references a child collection.
  - `child_collection.json`: The child collection that contains a multiscale image.
  - `resolved_collection.json`: Equivalent collection structure with the child collection resolved (i.e. inlined) within the parent collection.
- `webknossos/inline_multiscale.json`: A collection with a 3 multiscale images (EM, segmentation, prediction) and a custom nodes.
