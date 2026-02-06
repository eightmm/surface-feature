"""Surface vertex featurizer from PDB.

Single-file module that builds a molecular surface from a PDB file and
computes vertex-wise MaSIF-style features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
import torch
from rdkit import Chem
from scipy.spatial.distance import cdist
from skimage import measure
import trimesh


@dataclass(frozen=True)
class SurfaceFeatures:
    verts: np.ndarray  # (N, 3)
    faces: np.ndarray  # (F, 3)
    normals: np.ndarray  # (N, 3)
    features: Dict[str, np.ndarray]


RESIDUE_ATOM_TOKEN = {
    # ALA: N, CA, C, O, CB
    ("ALA", "N"): 0, ("ALA", "CA"): 1, ("ALA", "C"): 2, ("ALA", "O"): 3, ("ALA", "CB"): 4,
    # ARG: N, CA, C, O, CB, CG, CD, NE, CZ, NH1, NH2
    ("ARG", "N"): 5, ("ARG", "CA"): 6, ("ARG", "C"): 7, ("ARG", "O"): 8, ("ARG", "CB"): 9,
    ("ARG", "CG"): 10, ("ARG", "CD"): 11, ("ARG", "NE"): 12, ("ARG", "CZ"): 13,
    ("ARG", "NH1"): 14, ("ARG", "NH2"): 15,
    # ASN: N, CA, C, O, CB, CG, OD1, ND2
    ("ASN", "N"): 16, ("ASN", "CA"): 17, ("ASN", "C"): 18, ("ASN", "O"): 19,
    ("ASN", "CB"): 20, ("ASN", "CG"): 21, ("ASN", "OD1"): 22, ("ASN", "ND2"): 23,
    # ASP: N, CA, C, O, CB, CG, OD1, OD2
    ("ASP", "N"): 24, ("ASP", "CA"): 25, ("ASP", "C"): 26, ("ASP", "O"): 27,
    ("ASP", "CB"): 28, ("ASP", "CG"): 29, ("ASP", "OD1"): 30, ("ASP", "OD2"): 31,
    # CYS: N, CA, C, O, CB, SG
    ("CYS", "N"): 32, ("CYS", "CA"): 33, ("CYS", "C"): 34, ("CYS", "O"): 35,
    ("CYS", "CB"): 36, ("CYS", "SG"): 37,
    # GLN: N, CA, C, O, CB, CG, CD, OE1, NE2
    ("GLN", "N"): 38, ("GLN", "CA"): 39, ("GLN", "C"): 40, ("GLN", "O"): 41,
    ("GLN", "CB"): 42, ("GLN", "CG"): 43, ("GLN", "CD"): 44, ("GLN", "OE1"): 45,
    ("GLN", "NE2"): 46,
    # GLU: N, CA, C, O, CB, CG, CD, OE1, OE2
    ("GLU", "N"): 47, ("GLU", "CA"): 48, ("GLU", "C"): 49, ("GLU", "O"): 50,
    ("GLU", "CB"): 51, ("GLU", "CG"): 52, ("GLU", "CD"): 53, ("GLU", "OE1"): 54,
    ("GLU", "OE2"): 55,
    # GLY: N, CA, C, O
    ("GLY", "N"): 56, ("GLY", "CA"): 57, ("GLY", "C"): 58, ("GLY", "O"): 59,
    # HIS: N, CA, C, O, CB, CG, ND1, CD2, CE1, NE2
    ("HIS", "N"): 60, ("HIS", "CA"): 61, ("HIS", "C"): 62, ("HIS", "O"): 63,
    ("HIS", "CB"): 64, ("HIS", "CG"): 65, ("HIS", "ND1"): 66, ("HIS", "CD2"): 67,
    ("HIS", "CE1"): 68, ("HIS", "NE2"): 69,
    # ILE: N, CA, C, O, CB, CG1, CG2, CD1
    ("ILE", "N"): 70, ("ILE", "CA"): 71, ("ILE", "C"): 72, ("ILE", "O"): 73,
    ("ILE", "CB"): 74, ("ILE", "CG1"): 75, ("ILE", "CG2"): 76, ("ILE", "CD1"): 77,
    # LEU: N, CA, C, O, CB, CG, CD1, CD2
    ("LEU", "N"): 78, ("LEU", "CA"): 79, ("LEU", "C"): 80, ("LEU", "O"): 81,
    ("LEU", "CB"): 82, ("LEU", "CG"): 83, ("LEU", "CD1"): 84, ("LEU", "CD2"): 85,
    # LYS: N, CA, C, O, CB, CG, CD, CE, NZ
    ("LYS", "N"): 86, ("LYS", "CA"): 87, ("LYS", "C"): 88, ("LYS", "O"): 89,
    ("LYS", "CB"): 90, ("LYS", "CG"): 91, ("LYS", "CD"): 92, ("LYS", "CE"): 93,
    ("LYS", "NZ"): 94,
    # MET: N, CA, C, O, CB, CG, SD, CE
    ("MET", "N"): 95, ("MET", "CA"): 96, ("MET", "C"): 97, ("MET", "O"): 98,
    ("MET", "CB"): 99, ("MET", "CG"): 100, ("MET", "SD"): 101, ("MET", "CE"): 102,
    # PHE: N, CA, C, O, CB, CG, CD1, CD2, CE1, CE2, CZ
    ("PHE", "N"): 103, ("PHE", "CA"): 104, ("PHE", "C"): 105, ("PHE", "O"): 106,
    ("PHE", "CB"): 107, ("PHE", "CG"): 108, ("PHE", "CD1"): 109, ("PHE", "CD2"): 110,
    ("PHE", "CE1"): 111, ("PHE", "CE2"): 112, ("PHE", "CZ"): 113,
    # PRO: N, CA, C, O, CB, CG, CD
    ("PRO", "N"): 114, ("PRO", "CA"): 115, ("PRO", "C"): 116, ("PRO", "O"): 117,
    ("PRO", "CB"): 118, ("PRO", "CG"): 119, ("PRO", "CD"): 120,
    # SER: N, CA, C, O, CB, OG
    ("SER", "N"): 121, ("SER", "CA"): 122, ("SER", "C"): 123, ("SER", "O"): 124,
    ("SER", "CB"): 125, ("SER", "OG"): 126,
    # THR: N, CA, C, O, CB, OG1, CG2
    ("THR", "N"): 127, ("THR", "CA"): 128, ("THR", "C"): 129, ("THR", "O"): 130,
    ("THR", "CB"): 131, ("THR", "OG1"): 132, ("THR", "CG2"): 133,
    # TRP: N, CA, C, O, CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2
    ("TRP", "N"): 134, ("TRP", "CA"): 135, ("TRP", "C"): 136, ("TRP", "O"): 137,
    ("TRP", "CB"): 138, ("TRP", "CG"): 139, ("TRP", "CD1"): 140, ("TRP", "CD2"): 141,
    ("TRP", "NE1"): 142, ("TRP", "CE2"): 143, ("TRP", "CE3"): 144, ("TRP", "CZ2"): 145,
    ("TRP", "CZ3"): 146, ("TRP", "CH2"): 147,
    # TYR: N, CA, C, O, CB, CG, CD1, CD2, CE1, CE2, CZ, OH
    ("TYR", "N"): 148, ("TYR", "CA"): 149, ("TYR", "C"): 150, ("TYR", "O"): 151,
    ("TYR", "CB"): 152, ("TYR", "CG"): 153, ("TYR", "CD1"): 154, ("TYR", "CD2"): 155,
    ("TYR", "CE1"): 156, ("TYR", "CE2"): 157, ("TYR", "CZ"): 158, ("TYR", "OH"): 159,
    # VAL: N, CA, C, O, CB, CG1, CG2
    ("VAL", "N"): 160, ("VAL", "CA"): 161, ("VAL", "C"): 162, ("VAL", "O"): 163,
    ("VAL", "CB"): 164, ("VAL", "CG1"): 165, ("VAL", "CG2"): 166,
    # UNK: N, CA, C, O, CB (unknown residue, backbone + CB only)
    ("UNK", "N"): 167, ("UNK", "CA"): 168, ("UNK", "C"): 169, ("UNK", "O"): 170, ("UNK", "CB"): 171,
    ("XXX", "N"): 167, ("XXX", "CA"): 168, ("XXX", "C"): 169, ("XXX", "O"): 170, ("XXX", "CB"): 171,
    # Metal ions (biologically important metals with distinct roles)
    ("METAL", "CA"): 175,   # Calcium - signaling, structural
    ("METAL", "MG"): 176,   # Magnesium - enzymatic cofactor, ATP binding
    ("METAL", "ZN"): 177,   # Zinc - structural (zinc fingers), catalytic
    ("METAL", "FE"): 178,   # Iron - electron transfer, oxygen binding
    ("METAL", "MN"): 179,   # Manganese - photosynthesis, oxidoreductases
    ("METAL", "CU"): 180,   # Copper - electron transfer, oxidases
    ("METAL", "CO"): 181,   # Cobalt - vitamin B12, some enzymes
    ("METAL", "NI"): 182,   # Nickel - urease, hydrogenases
    ("METAL", "NA"): 183,   # Sodium - ion channels, osmotic balance
    ("METAL", "K"): 184,    # Potassium - ion channels, protein stability
    ("METAL", "METAL"): 185,  # Generic/unspecified metal
    # Special tokens
    ("UNK", "UNK"): 186,
}


def _mol_from_pdb(pdb_path: str) -> Chem.Mol:
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"Failed to read PDB: {pdb_path}")
    return mol


def _mol_positions(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    pos = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        pos[i] = [p.x, p.y, p.z]
    return pos


def _vdw_radii(mol: Chem.Mol) -> np.ndarray:
    pt = Chem.GetPeriodicTable()
    radii = np.zeros((mol.GetNumAtoms(),), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        r = pt.GetRvdw(atom.GetAtomicNum())
        if r <= 0:
            r = 1.5
        radii[i] = r
    return radii


def create_surface_mesh(
    positions: np.ndarray,
    radii: np.ndarray,
    grid_density: float = 2.5,
    threshold: float = 0.5,
    sharpness: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a surface mesh using marching cubes on a Gaussian field."""
    padding = 2.0
    min_bound = positions.min(axis=0) - padding
    max_bound = positions.max(axis=0) + padding
    size = max_bound - min_bound

    res_x, res_y, res_z = (int(s * grid_density) for s in size)
    res_x, res_y, res_z = max(res_x, 10), max(res_y, 10), max(res_z, 10)

    n_atoms = len(positions)
    n_grid_points = res_x * res_y * res_z

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Batch size targeting ~1GB memory per batch (conservative)
    bytes_per_atom = n_grid_points * 4 * 3  # float32, ~3 intermediate tensors
    batch_size = max(1, (1 * 1024**3) // bytes_per_atom)
    batch_size = min(batch_size, n_atoms)

    pos_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
    rad_tensor = torch.tensor(radii, dtype=torch.float32, device=device)

    min_bound_t = torch.tensor(min_bound, dtype=torch.float32, device=device)
    max_bound_t = torch.tensor(max_bound, dtype=torch.float32, device=device)

    x = torch.linspace(min_bound_t[0], max_bound_t[0], res_x, device=device)
    y = torch.linspace(min_bound_t[1], max_bound_t[1], res_y, device=device)
    z = torch.linspace(min_bound_t[2], max_bound_t[2], res_z, device=device)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    scalar_field = torch.zeros((res_x, res_y, res_z), dtype=torch.float32, device=device)

    for start_idx in range(0, n_atoms, batch_size):
        end_idx = min(start_idx + batch_size, n_atoms)
        batch_pos = pos_tensor[start_idx:end_idx]
        batch_rad = rad_tensor[start_idx:end_idx]

        diff = grid_coords.unsqueeze(0) - batch_pos.view(-1, 1, 1, 1, 3)
        dist_sq = torch.sum(diff**2, dim=-1)
        blobs = torch.exp(-sharpness * (dist_sq / (batch_rad.view(-1, 1, 1, 1) ** 2)))
        scalar_field += torch.sum(blobs, dim=0)

        del diff, dist_sq, blobs
        if device == "cuda":
            torch.cuda.empty_cache()

    scalar_field = scalar_field.cpu().numpy()

    verts, faces, normals, _ = measure.marching_cubes(scalar_field, level=threshold)
    scale = np.array(
        [
            size[0] / (res_x - 1),
            size[1] / (res_y - 1),
            size[2] / (res_z - 1),
        ]
    )
    verts = verts * scale + min_bound

    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    return verts, faces, normals


def _compute_mesh_surface_area(verts: np.ndarray, faces: np.ndarray) -> float:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return areas.sum()


def simplify_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    target_face_area: float = 1.0,
    min_faces: int = 100,
    max_faces: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simplify mesh using Open3D's Quadric Error Metrics (QEM) algorithm."""
    total_area = _compute_mesh_surface_area(verts, faces)
    target_faces = int(total_area / target_face_area)
    target_faces = max(target_faces, min_faces)
    if max_faces is not None:
        target_faces = min(target_faces, max_faces)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()

    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

    new_verts = np.asarray(simplified.vertices)
    new_faces = np.asarray(simplified.triangles)
    new_normals = np.asarray(simplified.vertex_normals)

    norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
    new_normals = new_normals / (norms + 1e-8)

    if len(new_verts) > 0:
        centroid = new_verts.mean(axis=0)
        outward_dir = new_verts - centroid
        dot_products = np.sum(new_normals * outward_dir, axis=1)
        flip_mask = dot_products < 0
        new_normals[flip_mask] *= -1

    return new_verts, new_faces, new_normals


def compute_all_vertex_features(
    verts: np.ndarray,
    faces: np.ndarray,
    atom_positions: np.ndarray,
    mol: Chem.Mol,
    feature_radii: Tuple[float, ...] = (2.0, 4.0, 6.0),
) -> Dict[str, np.ndarray]:
    """Compute protein surface features at each vertex (vertex-based)."""
    n_verts = len(verts)

    # Precompute normals/mesh
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        base_radius = 1.5

        mean_curv = trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=base_radius
        )
        gauss_curv = trimesh.curvature.discrete_gaussian_curvature_measure(
            mesh, mesh.vertices, radius=base_radius
        )

        k_sum = mean_curv * 2
        k_diff = np.sqrt(np.maximum(k_sum**2 - 4 * gauss_curv, 0))
        k_diff = np.maximum(k_diff, 1e-8)
        shape_idx = (2 / np.pi) * np.arctan2(k_sum, k_diff)
        shape_idx = np.clip(shape_idx, -1, 1)
    except Exception:
        shape_idx = np.zeros(n_verts)
        mean_curv = np.zeros(n_verts)
        gauss_curv = np.zeros(n_verts)

    charged = {
        "ASP": {"OD1": -0.5, "OD2": -0.5},
        "GLU": {"OE1": -0.5, "OE2": -0.5},
        "LYS": {"NZ": 1.0},
        "ARG": {"NH1": 0.5, "NH2": 0.5},
        "HIS": {"ND1": 0.5, "NE2": 0.5},
    }
    charges = np.zeros(mol.GetNumAtoms())
    for atom in mol.GetAtoms():
        res = atom.GetPDBResidueInfo()
        if res:
            res_name = res.GetResidueName().strip()
            atom_name = res.GetName().strip()
            if res_name in charged:
                charges[atom.GetIdx()] = charged[res_name].get(atom_name, 0.0)

    distances = cdist(verts, atom_positions)
    distances = np.maximum(distances, 0.5)
    mask = distances < 10.0
    electrostatic = np.sum(np.where(mask, charges / distances, 0), axis=1)
    electrostatic = np.clip(electrostatic, -2, 2)

    kd_scale = {
        "ILE": 4.5,
        "VAL": 4.2,
        "LEU": 3.8,
        "PHE": 2.8,
        "CYS": 2.5,
        "MET": 1.9,
        "ALA": 1.8,
        "GLY": -0.4,
        "THR": -0.7,
        "SER": -0.8,
        "TRP": -0.9,
        "TYR": -1.3,
        "PRO": -1.6,
        "HIS": -3.2,
        "GLU": -3.5,
        "GLN": -3.5,
        "ASP": -3.5,
        "ASN": -3.5,
        "LYS": -3.9,
        "ARG": -4.5,
    }
    logp_contribs = np.zeros(mol.GetNumAtoms())
    mr_contribs = np.zeros(mol.GetNumAtoms())
    hbd_atoms = np.zeros(mol.GetNumAtoms())
    hba_atoms = np.zeros(mol.GetNumAtoms())
    aromaticity_atoms = np.zeros(mol.GetNumAtoms())
    pos_atoms = np.zeros(mol.GetNumAtoms())
    neg_atoms = np.zeros(mol.GetNumAtoms())

    for atom in mol.GetAtoms():
        res = atom.GetPDBResidueInfo()
        if res:
            res_name = res.GetResidueName().strip()
            logp_contribs[atom.GetIdx()] = kd_scale.get(res_name, 0.0)
            if atom.GetAtomicNum() == 7:
                hbd_atoms[atom.GetIdx()] = 1.0
            if atom.GetAtomicNum() == 8:
                hba_atoms[atom.GetIdx()] = 1.0

    weights = 1.0 / (distances + 1e-8)
    weights = weights / weights.sum(axis=1, keepdims=True)

    hydrophobicity = np.clip((weights * logp_contribs).sum(axis=1), -5, 5)
    molar_refractivity = np.clip((weights * mr_contribs).sum(axis=1), 0, 10)
    hbd = np.clip((weights * hbd_atoms).sum(axis=1), 0, 1)
    hba = np.clip((weights * hba_atoms).sum(axis=1), 0, 1)
    aromaticity = np.clip((weights * aromaticity_atoms).sum(axis=1), 0, 1)
    pos_ionizable = np.clip((weights * pos_atoms).sum(axis=1), 0, 1)
    neg_ionizable = np.clip((weights * neg_atoms).sum(axis=1), 0, 1)

    vertex_normals = mesh.vertex_normals
    convexity = np.sign(mean_curv)

    # Protein-only local geometric features (no external tools)
    # These are still computed for ligand too (useful), but most meaningful for protein surfaces.
    from scipy.spatial import cKDTree

    tree = cKDTree(verts)
    # Per-vertex area (sum 1/3 of adjacent face areas)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    vertex_area = np.zeros(n_verts, dtype=np.float32)
    np.add.at(vertex_area, faces[:, 0], face_areas / 3.0)
    np.add.at(vertex_area, faces[:, 1], face_areas / 3.0)
    np.add.at(vertex_area, faces[:, 2], face_areas / 3.0)

    # Distance to centroid (simple global context)
    centroid = verts.mean(axis=0)
    dist_to_centroid = np.linalg.norm(verts - centroid, axis=1)

    # Multi-scale neighborhood stats
    local_stats: Dict[str, np.ndarray] = {}
    for r in feature_radii:
        # Indices for each vertex within radius r
        neighborhoods = tree.query_ball_point(verts, r)

        neighbor_count = np.array([len(nb) for nb in neighborhoods], dtype=np.float32)
        # Local area density: sum of vertex areas in neighborhood / neighborhood size
        local_area = np.zeros(n_verts, dtype=np.float32)
        normal_var = np.zeros(n_verts, dtype=np.float32)
        pca_eigvals = np.zeros((n_verts, 3), dtype=np.float32)
        pca_linearity = np.zeros(n_verts, dtype=np.float32)
        pca_planarity = np.zeros(n_verts, dtype=np.float32)
        pca_sphericity = np.zeros(n_verts, dtype=np.float32)

        for i, nb in enumerate(neighborhoods):
            if not nb:
                continue
            nb = np.array(nb, dtype=np.int32)
            local_area[i] = vertex_area[nb].sum()
            nbs = vertex_normals[nb]
            normal_var[i] = np.mean(np.var(nbs, axis=0))
            if nb.size >= 3:
                pts = verts[nb] - verts[i]
                cov = (pts.T @ pts) / max(1, pts.shape[0])
                w = np.linalg.eigvalsh(cov)
                w = np.sort(w)[::-1]
                s = w.sum()
                if s > 0:
                    w = w / s
                pca_eigvals[i] = w
                if w[0] > 0:
                    pca_linearity[i] = (w[0] - w[1]) / w[0]
                    pca_planarity[i] = (w[1] - w[2]) / w[0]
                    pca_sphericity[i] = w[2] / w[0]

        local_stats[f"neighbor_count_r{r}"] = neighbor_count
        local_stats[f"local_area_r{r}"] = local_area
        local_stats[f"normal_var_r{r}"] = normal_var
        local_stats[f"pca_eigvals_r{r}"] = pca_eigvals
        local_stats[f"pca_linearity_r{r}"] = pca_linearity
        local_stats[f"pca_planarity_r{r}"] = pca_planarity
        local_stats[f"pca_sphericity_r{r}"] = pca_sphericity

    # Multi-scale curvature (optional but cheap relative to surface gen)
    multi_curv: Dict[str, np.ndarray] = {}
    try:
        for r in feature_radii:
            m = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, radius=r)
            g = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius=r)
            multi_curv[f"mean_curvature_r{r}"] = m
            multi_curv[f"gaussian_curvature_r{r}"] = g
    except Exception:
        for r in feature_radii:
            multi_curv[f"mean_curvature_r{r}"] = np.zeros(n_verts)
            multi_curv[f"gaussian_curvature_r{r}"] = np.zeros(n_verts)

    aa_list = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
    residue_onehot = np.zeros((mol.GetNumAtoms(), 20))
    for atom in mol.GetAtoms():
        res = atom.GetPDBResidueInfo()
        if res:
            res_name = res.GetResidueName().strip()
            if res_name in aa_to_idx:
                residue_onehot[atom.GetIdx(), aa_to_idx[res_name]] = 1.0
    vertex_residue_type = weights @ residue_onehot

    # Residue-atom token one-hot (187)
    rat_tokens = _atom_residue_tokens(mol)
    rat_onehot = np.zeros((mol.GetNumAtoms(), 187), dtype=np.float32)
    rat_onehot[np.arange(mol.GetNumAtoms()), rat_tokens] = 1.0
    vertex_residue_atom_type = weights @ rat_onehot

    backbone_names = {"N", "CA", "C", "O"}
    is_backbone = np.zeros(mol.GetNumAtoms())
    for atom in mol.GetAtoms():
        res = atom.GetPDBResidueInfo()
        if res and res.GetName().strip() in backbone_names:
            is_backbone[atom.GetIdx()] = 1.0
    vertex_backbone = (weights * is_backbone).sum(axis=1)

    b_factors = np.zeros(mol.GetNumAtoms())
    for atom in mol.GetAtoms():
        res = atom.GetPDBResidueInfo()
        if res:
            b_factors[atom.GetIdx()] = res.GetTempFactor()
    if b_factors.max() > 0:
        b_factors = b_factors / b_factors.max()
    vertex_bfactor = (weights * b_factors).sum(axis=1)

    return {
        "shape_index": shape_idx,
        "mean_curvature": mean_curv,
        "gaussian_curvature": gauss_curv,
        "electrostatic": electrostatic,
        "hydrophobicity": hydrophobicity,
        "hbd": hbd,
        "hba": hba,
        "molar_refractivity": molar_refractivity,
        "aromaticity": aromaticity,
        "pos_ionizable": pos_ionizable,
        "neg_ionizable": neg_ionizable,
        "vertex_normal": vertex_normals,
        "convexity": convexity,
        "atom_type": vertex_atom_type,
        "residue_type": vertex_residue_type,
        "residue_atom_type": vertex_residue_atom_type,
        "is_backbone": vertex_backbone,
        "b_factor": vertex_bfactor,
        "distance_to_centroid": dist_to_centroid,
        **local_stats,
        **multi_curv,
    }


def extract_surface_vertex_features_from_pdb(
    pdb_path: str,
    *,
    grid_density: float = 2.5,
    threshold: float = 0.5,
    sharpness: float = 1.5,
    simplify: bool = True,
    target_face_area: float = 1.0,
    min_faces: int = 100,
    max_faces: int | None = None,
    feature_radii: Tuple[float, ...] = (2.0, 4.0, 6.0),
) -> SurfaceFeatures:
    """Build surface and compute vertex features from a PDB file."""
    mol = _mol_from_pdb(pdb_path)
    positions = _mol_positions(mol)
    radii = _vdw_radii(mol)

    verts, faces, normals = create_surface_mesh(
        positions,
        radii,
        grid_density=grid_density,
        threshold=threshold,
        sharpness=sharpness,
    )

    if verts is None or len(verts) == 0:
        raise RuntimeError("Surface generation failed or produced empty mesh")

    if simplify:
        verts, faces, normals = simplify_mesh(
            verts,
            faces,
            normals,
            target_face_area=target_face_area,
            min_faces=min_faces,
            max_faces=max_faces,
        )

    features = compute_all_vertex_features(
        verts,
        faces,
        positions,
        mol,
        feature_radii=feature_radii,
    )

    return SurfaceFeatures(verts=verts, faces=faces, normals=normals, features=features)
def _residue_atom_token(resname: str, atom_name: str, element: str) -> int:
    res = (resname or "UNK").strip().upper()
    atom = (atom_name or "UNK").strip().upper()
    elem = (element or "").strip().upper()

    tok = RESIDUE_ATOM_TOKEN.get((res, atom), None)
    if tok is not None:
        return tok

    if elem in {"CA", "MG", "ZN", "FE", "MN", "CU", "CO", "NI", "NA", "K"}:
        tok = RESIDUE_ATOM_TOKEN.get(("METAL", elem), None)
        if tok is not None:
            return tok

    tok = RESIDUE_ATOM_TOKEN.get(("UNK", atom), None)
    if tok is not None:
        return tok

    return RESIDUE_ATOM_TOKEN[("UNK", "UNK")]


def _atom_residue_tokens(mol: Chem.Mol) -> np.ndarray:
    tokens = np.zeros(mol.GetNumAtoms(), dtype=np.int64)
    for atom in mol.GetAtoms():
        res = atom.GetPDBResidueInfo()
        if res:
            resname = res.GetResidueName().strip()
            atom_name = res.GetName().strip()
            elem = atom.GetSymbol().strip().upper()
            tokens[atom.GetIdx()] = _residue_atom_token(resname, atom_name, elem)
        else:
            elem = atom.GetSymbol().strip().upper()
            tokens[atom.GetIdx()] = _residue_atom_token("UNK", "UNK", elem)
    return tokens
