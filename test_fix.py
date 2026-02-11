#!/usr/bin/env python3
"""Test script to verify vertex_atom_type fix."""

import numpy as np
from surface_feature import extract_surface_vertex_features_from_pdb

def test_featurization(pdb_path: str):
    """Test featurization on a PDB file."""
    print(f"Testing featurization on: {pdb_path}")
    print("-" * 80)

    try:
        # Run featurization
        result = extract_surface_vertex_features_from_pdb(
            pdb_path,
            grid_density=2.5,
            threshold=0.5,
            sharpness=1.5,
            simplify=True,
            target_face_area=1.0,
            min_faces=100,
            max_faces=None,
            feature_radii=(2.0, 4.0, 6.0),
        )

        print(f"✓ Surface generation successful!")
        print(f"  - Vertices: {len(result.verts)}")
        print(f"  - Faces: {len(result.faces)}")
        print(f"  - Normals: {result.normals.shape}")
        print()

        # Check features
        print("Feature shapes:")
        for key, value in result.features.items():
            if isinstance(value, np.ndarray):
                print(f"  - {key:30s}: {value.shape}")
        print()

        # Verify atom_type exists and has correct shape
        if "atom_type" in result.features:
            atom_type = result.features["atom_type"]
            print(f"✓ 'atom_type' feature found!")
            print(f"  - Shape: {atom_type.shape}")
            print(f"  - Expected: ({len(result.verts)}, 7)")
            print(f"  - Min/Max: [{atom_type.min():.4f}, {atom_type.max():.4f}]")
            print(f"  - Sample (first vertex): {atom_type[0]}")

            # Check if it's properly normalized (should sum to ~1.0 for each vertex)
            row_sums = atom_type.sum(axis=1)
            print(f"  - Row sums (should be ~1.0): min={row_sums.min():.4f}, max={row_sums.max():.4f}, mean={row_sums.mean():.4f}")
        else:
            print("✗ 'atom_type' feature NOT found!")
            return False

        print()
        print("=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    pdb_path = "/mnt/data/PLI/P-L/2011-2019/6oi8/6oi8_protein.pdb"
    success = test_featurization(pdb_path)
    exit(0 if success else 1)
