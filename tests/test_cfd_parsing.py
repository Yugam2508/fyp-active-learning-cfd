"""Unit tests for the parts of src/cfd_dispersion.py that do not need OpenFOAM."""
import numpy as np
import cfd_dispersion as cfd


def test_levels_recovers_tensor_grid_with_rounding_noise():
    xs, zs = np.linspace(10, 250, 7), np.linspace(0.3, 9.7, 5)
    Xg, Zg = np.meshgrid(xs, zs, indexing="ij")
    noise = np.random.default_rng(0).normal(scale=1e-9, size=Xg.size)
    lx, ix = cfd._levels(Xg.ravel() + noise)
    lz, iz = cfd._levels(Zg.ravel() - noise)
    assert np.allclose(lx, xs) and np.allclose(lz, zs)
    assert len(set(zip(ix, iz))) == Xg.size


def test_receptor_concentration_is_linear_in_release_rate():
    prof = {"x": [5.0, 50.0, 120.0], "s": [0.3, 0.2, 0.1]}
    c1 = cfd.receptor_concentration(prof, 27.5, 1.0)
    assert np.isclose(c1, 1e3 * 0.25)
    assert np.isclose(cfd.receptor_concentration(prof, 27.5, 7.0), 7 * c1)


def test_block_mesh_has_expected_cell_count():
    txt = cfd._block_mesh(10.0)
    cells = [tuple(map(int, l.split("(")[2].split(")")[0].split())) for l in txt.splitlines()
             if l.strip().startswith("hex")]
    assert sum(a * b * c for a, b, c in cells) == 7776
