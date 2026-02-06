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

Scalar features (one value per vertex):

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
- `residue_type` (N, 20)
- `residue_atom_type` (N, 187)

### Feature Descriptions

- `shape_index`: normalized local shape descriptor from principal curvatures in `[-1, 1]`.
- `mean_curvature`: mean curvature at each vertex.
- `gaussian_curvature`: Gaussian curvature at each vertex.
- `electrostatic`: charge/distance weighted potential (clipped to `[-2, 2]`).
- `hydrophobicity`: residue-based hydrophobicity (protein).
- `hbd`: hydrogen bond donor likelihood (0-1).
- `hba`: hydrogen bond acceptor likelihood (0-1).
- `molar_refractivity`: residue-level refractivity proxy (protein).
- `aromaticity`: aromatic atom density (0-1).
- `pos_ionizable`: positively ionizable density (0-1).
- `neg_ionizable`: negatively ionizable density (0-1).
- `convexity`: sign of mean curvature (concave/convex indicator).
- `is_backbone`: backbone atom density (protein only).
- `b_factor`: normalized B-factor density (protein only).
- `distance_to_centroid`: Euclidean distance from vertex to surface centroid.
- `neighbor_count_r{r}`: number of vertices within radius `r`.
- `local_area_r{r}`: sum of vertex areas within radius `r`.
- `normal_var_r{r}`: variance of vertex normals within radius `r`.
- `pca_eigvals_r{r}`: normalized eigenvalues of local covariance (3 values).
- `pca_linearity_r{r}`: `(λ1-λ2)/λ1` from local PCA.
- `pca_planarity_r{r}`: `(λ2-λ3)/λ1` from local PCA.
- `pca_sphericity_r{r}`: `λ3/λ1` from local PCA.
- `mean_curvature_r{r}`: mean curvature measured at radius `r`.
- `gaussian_curvature_r{r}`: Gaussian curvature measured at radius `r`.
- `vertex_normal`: unit normal vector at vertex.
- `residue_type`: one-hot amino acid type (protein only).
- `residue_atom_type`: one-hot residue+atom token (protein only, 187 classes).

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
