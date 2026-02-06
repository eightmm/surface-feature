# surface-feature

Single-file PDB protein surface featurizer. It builds a protein surface from a PDB file and computes vertex-wise features.

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
)

verts = result.verts
faces = result.faces
features = result.features
```

### Accessing Feature Arrays

```python
from surface_feature import extract_surface_vertex_features_from_pdb

result = extract_surface_vertex_features_from_pdb("path/to/protein.pdb")

# Example: scalar feature (N,)
shape_index = result.features["shape_index"]

# Example: vector feature (N, 3)
normals = result.features["vertex_normal"]

# Example: one-hot feature (N, 20)
res_type = result.features["residue_type"]

# Example: PCA eigenvalues at radius 4.0 (N, 3)
pca_eigs_r4 = result.features["pca_eigvals_r4.0"]
```

## Options

All options are keyword-only in `extract_surface_vertex_features_from_pdb`.

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `grid_density` | `float` | `2.5` | Grid points per Angstrom for marching cubes. Higher = finer surface, slower and higher memory use. |
| `threshold` | `float` | `0.5` | Isosurface threshold in the scalar field. Lower = larger surface, higher = tighter surface. |
| `sharpness` | `float` | `1.5` | Controls Gaussian sharpness. Higher = sharper surface closer to VdW radii. |
| `simplify` | `bool` | `True` | Whether to simplify the mesh using QEM. |
| `target_face_area` | `float` | `1.0` | Target face area in Å² when simplifying. Lower = more faces. |
| `min_faces` | `int` | `100` | Minimum faces after simplification (safety floor). |
| `max_faces` | `int \| None` | `None` | Cap faces after simplification (safety ceiling). |
| `feature_radii` | `tuple[float, ...]` | `(2.0, 4.0, 6.0)` | Radii (Å) for multi-scale curvature and local neighborhood stats. Larger radii capture global shape, smaller radii capture local detail. |

Example with tuned mesh resolution:

```python
result = extract_surface_vertex_features_from_pdb(
    "path/to/protein.pdb",
    grid_density=3.0,
    sharpness=2.0,
    simplify=True,
    target_face_area=0.7,
    max_faces=50000,
    feature_radii=(2.0, 4.0),
)
```

## Option Interactions

- `simplify=False`: `target_face_area`, `min_faces`, and `max_faces` are ignored because the mesh is not simplified.
- `grid_density` vs `target_face_area`: `grid_density` controls surface sampling during marching cubes; `target_face_area` controls post-simplification density. If you increase `grid_density`, you can usually increase `target_face_area` to keep runtime reasonable.
- `threshold` vs `sharpness`: higher `sharpness` makes the scalar field steeper, so small changes in `threshold` have stronger effects. If you increase `sharpness`, keep `threshold` near `0.5`.
- `feature_radii`: affects only local statistics and multi-scale curvature. It does not change the mesh itself.

None of the options are mutually exclusive; they are designed to compose. The only “ignore” behavior is when `simplify=False`.

## Protein-Only Recommended Settings

These defaults work well for protein surface representation without external tools:

```python
result = extract_surface_vertex_features_from_pdb(
    "path/to/protein.pdb",
    grid_density=2.5,
    sharpness=1.5,
    simplify=True,
    target_face_area=1.0,
    feature_radii=(2.0, 4.0, 6.0),
)
```

If you need more surface detail, increase `grid_density` and lower `target_face_area`.

## Feature Keys

All features are vertex-aligned. Shapes below are per-vertex unless noted.

| Key | Shape | Description |
| --- | --- | --- |
| `shape_index` | `(N,)` | Normalized local shape descriptor from principal curvatures in `[-1, 1]`. |
| `mean_curvature` | `(N,)` | Mean curvature at each vertex. |
| `gaussian_curvature` | `(N,)` | Gaussian curvature at each vertex. |
| `electrostatic` | `(N,)` | Charge/distance weighted potential (clipped to `[-2, 2]`). |
| `hydrophobicity` | `(N,)` | Residue-based hydrophobicity. |
| `hbd` | `(N,)` | Hydrogen bond donor likelihood (0-1). |
| `hba` | `(N,)` | Hydrogen bond acceptor likelihood (0-1). |
| `molar_refractivity` | `(N,)` | Residue-level refractivity proxy. |
| `aromaticity` | `(N,)` | Aromatic atom density (0-1). |
| `pos_ionizable` | `(N,)` | Positively ionizable density (0-1). |
| `neg_ionizable` | `(N,)` | Negatively ionizable density (0-1). |
| `convexity` | `(N,)` | Sign of mean curvature (concave/convex indicator). |
| `is_backbone` | `(N,)` | Backbone atom density. |
| `b_factor` | `(N,)` | Normalized B-factor density. |
| `distance_to_centroid` | `(N,)` | Distance from vertex to surface centroid. |
| `neighbor_count_r{r}` | `(N,)` | Number of vertices within radius `r`. |
| `local_area_r{r}` | `(N,)` | Sum of vertex areas within radius `r`. |
| `normal_var_r{r}` | `(N,)` | Variance of vertex normals within radius `r`. |
| `pca_eigvals_r{r}` | `(N, 3)` | Normalized eigenvalues of local covariance. |
| `pca_linearity_r{r}` | `(N,)` | `(λ1-λ2)/λ1` from local PCA. |
| `pca_planarity_r{r}` | `(N,)` | `(λ2-λ3)/λ1` from local PCA. |
| `pca_sphericity_r{r}` | `(N,)` | `λ3/λ1` from local PCA. |
| `mean_curvature_r{r}` | `(N,)` | Mean curvature measured at radius `r`. |
| `gaussian_curvature_r{r}` | `(N,)` | Gaussian curvature measured at radius `r`. |
| `vertex_normal` | `(N, 3)` | Unit normal vector at vertex. |
| `residue_type` | `(N, 20)` | One-hot amino acid type. |
| `residue_atom_type` | `(N, 187)` | One-hot residue+atom token. |

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
