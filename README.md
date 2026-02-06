# surface-feature

Single-file PDB surface featurizer. It builds a molecular surface from a PDB file and computes vertex-wise MaSIF-style features.

## Install (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```python
from surface_feature import extract_surface_vertex_features_from_pdb

result = extract_surface_vertex_features_from_pdb(
    "path/to/protein.pdb",
    is_ligand=False,
)

verts = result.verts
faces = result.faces
features = result.features
```

## Dependencies

This module requires the following imports:

- `numpy`
- `open3d`
- `torch`
- `rdkit`
- `scipy`
- `scikit-image`
- `trimesh`

These are declared in `pyproject.toml`.
