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

## Options

All options are keyword-only in `extract_surface_vertex_features_from_pdb`.

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `is_ligand` | `bool` | `False` | If `True`, uses ligand feature logic (Gasteiger charges, RDKit chemical features). If `False`, uses protein logic (residue-based features). |
| `grid_density` | `float` | `2.5` | Grid points per Angstrom for marching cubes. Higher = finer surface, slower. |
| `threshold` | `float` | `0.5` | Isosurface threshold in the scalar field. |
| `sharpness` | `float` | `1.5` | Controls Gaussian sharpness. Higher = sharper surface closer to VdW radii. |
| `simplify` | `bool` | `True` | Whether to simplify the mesh using QEM. |
| `target_face_area` | `float` | `1.0` | Target face area in Å² when simplifying. Lower = more faces. |
| `min_faces` | `int` | `100` | Minimum faces after simplification. |
| `max_faces` | `int \| None` | `None` | Cap faces after simplification. |
| `feature_radii` | `tuple[float, ...]` | `(2.0, 4.0, 6.0)` | Radii (Å) for multi-scale curvature and local neighborhood stats. |

Example with tuned mesh resolution:

```python
result = extract_surface_vertex_features_from_pdb(
    "path/to/protein.pdb",
    is_ligand=False,
    grid_density=3.0,
    sharpness=2.0,
    simplify=True,
    target_face_area=0.7,
    max_faces=50000,
    feature_radii=(2.0, 4.0),
)
```

## Protein-Only Recommended Settings

These defaults work well for protein surface representation without external tools:

```python
result = extract_surface_vertex_features_from_pdb(
    "path/to/protein.pdb",
    is_ligand=False,
    grid_density=2.5,
    sharpness=1.5,
    simplify=True,
    target_face_area=1.0,
    feature_radii=(2.0, 4.0, 6.0),
)
```

If you need more surface detail, increase `grid_density` and lower `target_face_area`.

## Feature Keys

Scalar features:

- `shape_index`
- `mean_curvature`
- `gaussian_curvature`
- `electrostatic`
- `hydrophobicity`
- `hbd`
- `hba`
- `molar_refractivity`
- `aromaticity`
- `pos_ionizable`
- `neg_ionizable`
- `convexity`
- `hybridization`
- `in_ring`
- `ring_size`
- `is_backbone`
- `b_factor`
- `distance_to_centroid`
- `neighbor_count_r{r}`
- `local_area_r{r}`
- `normal_var_r{r}`
- `pca_eigvals_r{r}` (N, 3)
- `pca_linearity_r{r}`
- `pca_planarity_r{r}`
- `pca_sphericity_r{r}`
- `mean_curvature_r{r}`
- `gaussian_curvature_r{r}`

Vector / one-hot features:

- `vertex_normal` (N, 3)
- `atom_type` (N, 6)
- `residue_type` (N, 20)

## More MaSIF-like Additions (Suggestions)

If you want this closer to MaSIF v1/v2 behavior, these are the highest value additions:

- Geodesic patch sampling around each vertex (radius R) and compute aggregated descriptors per patch.
- Use mesh geodesic distances instead of Euclidean distances when building local neighborhoods.
- Compute curvature at multiple radii (multi-scale) for better local shape context.
- Add distance-to-pocket-center or distance-to-ligand-surface features for binding site tasks.
- Add surface roughness or local area density within a geodesic ball.
- Add atom-level electrostatics via Poisson-Boltzmann or APBS-derived grids (optional).

If you want any of these, tell me which items to implement first and the target runtime constraints.

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
