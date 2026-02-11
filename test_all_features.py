#!/usr/bin/env python3
"""Comprehensive test to verify all documented features."""

import numpy as np
from surface_feature import extract_surface_vertex_features_from_pdb

def test_all_features(pdb_path: str, feature_radii=(2.0, 4.0, 6.0)):
    """Test that all documented features are present with correct shapes."""
    print("=" * 80)
    print("COMPREHENSIVE FEATURE VALIDATION")
    print("=" * 80)
    print(f"\nPDB: {pdb_path}")
    print(f"Feature radii: {feature_radii}\n")

    # Run featurization
    result = extract_surface_vertex_features_from_pdb(
        pdb_path,
        feature_radii=feature_radii,
    )

    N = len(result.verts)
    print(f"Number of vertices (N): {N}\n")

    # Define expected features
    expected_features = {
        # Basic features (N,)
        "shape_index": (N,),
        "mean_curvature": (N,),
        "gaussian_curvature": (N,),
        "electrostatic": (N,),
        "hydrophobicity": (N,),
        "hbd": (N,),
        "hba": (N,),
        "molar_refractivity": (N,),
        "aromaticity": (N,),
        "pos_ionizable": (N,),
        "neg_ionizable": (N,),
        "convexity": (N,),
        "is_backbone": (N,),
        "b_factor": (N,),
        "distance_to_centroid": (N,),

        # Multi-dimensional features
        "vertex_normal": (N, 3),
        "residue_type": (N, 20),
        "residue_atom_type": (N, 187),
        "atom_type": (N, 7),  # Added feature
    }

    # Multi-scale features for each radius
    for r in feature_radii:
        expected_features.update({
            f"neighbor_count_r{r}": (N,),
            f"local_area_r{r}": (N,),
            f"normal_var_r{r}": (N,),
            f"pca_eigvals_r{r}": (N, 3),
            f"pca_linearity_r{r}": (N,),
            f"pca_planarity_r{r}": (N,),
            f"pca_sphericity_r{r}": (N,),
            f"mean_curvature_r{r}": (N,),
            f"gaussian_curvature_r{r}": (N,),
        })

    # Validate each feature
    print("-" * 80)
    print("FEATURE VALIDATION")
    print("-" * 80)

    missing_features = []
    wrong_shape_features = []
    correct_features = []

    for feature_name, expected_shape in sorted(expected_features.items()):
        if feature_name not in result.features:
            missing_features.append(feature_name)
            print(f"✗ {feature_name:35s} MISSING")
        else:
            actual_shape = result.features[feature_name].shape
            if actual_shape == expected_shape:
                correct_features.append(feature_name)
                print(f"✓ {feature_name:35s} {str(actual_shape):20s}")
            else:
                wrong_shape_features.append((feature_name, expected_shape, actual_shape))
                print(f"✗ {feature_name:35s} Expected: {expected_shape}, Got: {actual_shape}")

    # Check for unexpected features
    unexpected_features = set(result.features.keys()) - set(expected_features.keys())

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Correct features: {len(correct_features)}/{len(expected_features)}")

    if missing_features:
        print(f"\n✗ Missing features ({len(missing_features)}):")
        for name in missing_features:
            print(f"  - {name}")

    if wrong_shape_features:
        print(f"\n✗ Wrong shape features ({len(wrong_shape_features)}):")
        for name, expected, actual in wrong_shape_features:
            print(f"  - {name}: expected {expected}, got {actual}")

    if unexpected_features:
        print(f"\n⚠ Unexpected features ({len(unexpected_features)}):")
        for name in sorted(unexpected_features):
            print(f"  - {name}: {result.features[name].shape}")

    # Detailed statistics for key features
    print("\n" + "=" * 80)
    print("SAMPLE FEATURE STATISTICS")
    print("=" * 80)

    sample_features = [
        "shape_index",
        "electrostatic",
        "hydrophobicity",
        "atom_type",
        "residue_type",
        "residue_atom_type",
    ]

    for name in sample_features:
        if name in result.features:
            data = result.features[name]
            print(f"\n{name}:")
            print(f"  Shape: {data.shape}")
            print(f"  Range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"  Mean: {data.mean():.4f}")
            if len(data.shape) > 1:
                print(f"  Sample (first row): {data[0][:5]}...")  # First 5 elements
                if data.shape[1] <= 10:
                    row_sums = data.sum(axis=1)
                    print(f"  Row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f}, mean={row_sums.mean():.4f}")

    # Final verdict
    print("\n" + "=" * 80)
    success = (len(missing_features) == 0 and
               len(wrong_shape_features) == 0 and
               len(correct_features) == len(expected_features))

    if success:
        print("✓✓✓ ALL FEATURES VALIDATED SUCCESSFULLY! ✓✓✓")
    else:
        print("✗✗✗ VALIDATION FAILED ✗✗✗")
    print("=" * 80)

    return success


if __name__ == "__main__":
    pdb_path = "/mnt/data/PLI/P-L/2011-2019/6oi8/6oi8_protein.pdb"
    success = test_all_features(pdb_path)
    exit(0 if success else 1)
