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
from rdkit.Chem import AllChem, Crippen, Lipinski
from scipy.spatial.distance import cdist
from skimage import measure
import trimesh


@dataclass(frozen=True)
class SurfaceFeatures:
    verts: np.ndarray  # (N, 3)
    faces: np.ndarray  # (F, 3)
    normals: np.ndarray  # (N, 3)
    features: Dict[str, np.ndarray]


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
    is_ligand: bool = True,
    feature_radii: Tuple[float, ...] = (2.0, 4.0, 6.0),
) -> Dict[str, np.ndarray]:
    """Compute MaSIF-style surface features at each vertex (vertex-based)."""
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

    if is_ligand:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            charge = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
            if np.isnan(charge) or np.isinf(charge):
                charge = 0.0
            charges.append(charge)
        charges = np.array(charges)
    else:
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

    if is_ligand:
        contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
        logp_contribs = np.array([c[0] for c in contribs])
        mr_contribs = np.array([c[1] for c in contribs])

        hbd_atoms = np.zeros(mol.GetNumAtoms())
        for match in mol.GetSubstructMatches(Lipinski.HDonorSmarts):
            hbd_atoms[match[0]] = 1.0

        hba_atoms = np.zeros(mol.GetNumAtoms())
        for match in mol.GetSubstructMatches(Lipinski.HAcceptorSmarts):
            hba_atoms[match[0]] = 1.0

        aromaticity_atoms = np.array([1.0 if a.GetIsAromatic() else 0.0 for a in mol.GetAtoms()])

        pos_smarts = Chem.MolFromSmarts(
            "[+1,+2,$([NH2]-C(=N)N),$([NH]=C(N)N),$([nH]1ccnc1)]"
        )
        pos_atoms = np.zeros(mol.GetNumAtoms())
        for match in mol.GetSubstructMatches(pos_smarts):
            pos_atoms[match[0]] = 1.0

        neg_smarts = Chem.MolFromSmarts(
            "[-1,-2,$([CX3](=O)[OH]),$([CX3](=O)[O-]),$([SX4](=O)(=O)[OH])]"
        )
        neg_atoms = np.zeros(mol.GetNumAtoms())
        for match in mol.GetSubstructMatches(neg_smarts):
            neg_atoms[match[0]] = 1.0
    else:
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

    if is_ligand:
        atom_type_map = {6: 0, 7: 1, 8: 2, 16: 3, 9: 4, 17: 4, 35: 4, 53: 4}
        atom_types = np.array([atom_type_map.get(a.GetAtomicNum(), 5) for a in mol.GetAtoms()])
        atom_type_onehot = np.zeros((mol.GetNumAtoms(), 6))
        for i, t in enumerate(atom_types):
            atom_type_onehot[i, t] = 1.0
        vertex_atom_type = weights @ atom_type_onehot

        hyb_map = {
            Chem.HybridizationType.SP: 1,
            Chem.HybridizationType.SP2: 2,
            Chem.HybridizationType.SP3: 3,
        }
        hybridization = np.array([hyb_map.get(a.GetHybridization(), 0) for a in mol.GetAtoms()])
        vertex_hybridization = (weights * hybridization).sum(axis=1)

        ring_atoms = np.array([1.0 if a.IsInRing() else 0.0 for a in mol.GetAtoms()])
        vertex_ring = (weights * ring_atoms).sum(axis=1)

        ring_info = mol.GetRingInfo()
        ring_size = np.zeros(mol.GetNumAtoms())
        for atom in mol.GetAtoms():
            sizes = [len(r) for r in ring_info.AtomRings() if atom.GetIdx() in r]
            ring_size[atom.GetIdx()] = max(sizes) if sizes else 0
        vertex_ring_size = (weights * ring_size).sum(axis=1)

        vertex_residue_type = np.zeros((n_verts, 20))
        vertex_backbone = np.zeros(n_verts)
        vertex_bfactor = np.zeros(n_verts)
    else:
        vertex_atom_type = np.zeros((n_verts, 6))
        vertex_hybridization = np.zeros(n_verts)
        vertex_ring = np.zeros(n_verts)
        vertex_ring_size = np.zeros(n_verts)

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
        "hybridization": vertex_hybridization,
        "in_ring": vertex_ring,
        "ring_size": vertex_ring_size,
        "residue_type": vertex_residue_type,
        "is_backbone": vertex_backbone,
        "b_factor": vertex_bfactor,
        "distance_to_centroid": dist_to_centroid,
        **local_stats,
        **multi_curv,
    }


def extract_surface_vertex_features_from_pdb(
    pdb_path: str,
    *,
    is_ligand: bool = False,
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
        is_ligand=is_ligand,
        feature_radii=feature_radii,
    )

    return SurfaceFeatures(verts=verts, faces=faces, normals=normals, features=features)
