"""Tests for the phase-dispersion metrics of mstar_stats_summary.py (synthetic data)."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mstar_stats_summary as mss  # noqa: E402

pv = pytest.importorskip("pyvista")


def write_table(path: Path, header: list[str], columns: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for row in np.column_stack(columns):
            fh.write("\t".join(f"{v:.9g}" for v in row) + "\n")


def by_variable(rows: list[dict]) -> dict[str, dict]:
    return {r["variable"]: r for r in rows}


# ---------------------------------------------------------------- (a) interface stats

FLUID1_VOL, FLUID2_VOL = 0.02, 0.01


@pytest.fixture
def stats_case(tmp_path):
    stats = tmp_path / "case" / "Stats"
    t = np.arange(0.0, 4.0001, 0.1)
    write_table(stats / "Fluid.txt",
                ["Time [s]", "Fluid Volume [m^3]", "Mean Velocity [m/s]",
                 "Mean Energy Dissipation Rate [W/kg]",
                 "Mean Time-Avg Energy Dissipation Rate [W/kg]",
                 "Fluid 1 (Water) Volume [m^3]", "Fluid 1 (Water) Volume LB [m^3]",
                 "Fluid 2 (Oil) Volume [m^3]"],
                [t, np.full_like(t, 1e-3), np.full_like(t, 0.1),
                 np.where(t >= 1, 1.0, 5.0), np.full_like(t, 99.0),
                 np.full_like(t, FLUID1_VOL), np.full_like(t, 7.0),
                 np.full_like(t, FLUID2_VOL)])
    write_table(stats / "Interface.txt",
                ["Time [s]", "Area [m^2]", "Velocity Mean [m/s]"],
                [t, np.where(t >= 1, 3e-3, 1e-3), np.zeros_like(t)])
    return tmp_path / "case", stats


def test_interface_metrics_known_area(stats_case):
    _, stats = stats_case
    files = mss.find_stats_files(stats)
    rows, series = mss.interface_metrics(files, stats, 1.0, phi=0.2, v_liq=None)
    r = by_variable(rows)
    a = r["Specific Interfacial Area [1/m]"]
    assert a["mean"] == pytest.approx(3.0)
    assert a["std"] == pytest.approx(0.0, abs=1e-12)
    assert a["steady_method"] == "interface"
    assert a["n_samples"] == 31
    assert r["Sauter Diameter d32 (interface) [m]"]["mean"] == pytest.approx(6 * 0.2 / 3.0)
    assert series["a"][0] == pytest.approx(1.0)
    assert series["area"][-1] == pytest.approx(3e-3)


def test_fluid_volumes(stats_case):
    _, stats = stats_case
    fl = mss.fluid_volumes(mss.find_stats_files(stats))
    assert [(i, name) for i, name, *_ in fl] == [(1, "Water"), (2, "Oil")]
    assert fl[0][3][0] == pytest.approx(FLUID1_VOL)


def test_input_voxel_size(tmp_path):
    (tmp_path / "input.xml").write_text("<a>\n  <dx>0.00072000066</dx>\n</a>")
    (tmp_path / "out" / "Stats").mkdir(parents=True)
    assert mss.input_voxel_size(tmp_path) == pytest.approx(7.2000066e-4)
    assert mss.input_voxel_size(tmp_path / "out" / "Stats") == pytest.approx(7.2000066e-4)
    assert mss.input_voxel_size(tmp_path / "out" / "Stats" / "x" / "y") is None


def test_interface_metrics_liquid_volume_override_and_no_phi(stats_case, capsys):
    _, stats = stats_case
    files = mss.find_stats_files(stats)
    rows, series = mss.interface_metrics(files, stats, 1.0, phi=None, v_liq=2e-3)
    assert [r["variable"] for r in rows] == ["Specific Interfacial Area [1/m]"]
    assert rows[0]["mean"] == pytest.approx(1.5)
    assert "d32" not in series
    assert "--dispersed-fraction not given" in capsys.readouterr().err


def test_interface_metrics_missing_area_lists_headers(tmp_path, capsys):
    stats = tmp_path / "Stats"
    t = np.arange(0, 1, 0.1)
    write_table(stats / "Interface.txt", ["Time [s]", "Foo [m]"], [t, t])
    rows, series = mss.interface_metrics(mss.find_stats_files(stats), stats, 0, 0.2, 1e-3)
    assert rows == [] and series == {}
    assert "Foo [m]" in capsys.readouterr().err


def test_resolution_check(stats_case):
    _, stats = stats_case
    rows = mss.resolution_check(mss.find_stats_files(stats), 1.0, 1e-4,
                                sigma=0.05, rho_c=1000.0)
    r = by_variable(rows)
    d_h = 0.725 * (0.05 / 1000.0) ** 0.6 * 1.0 ** -0.4  # instantaneous eps = 1 W/kg
    assert r["Hinze Max Stable Drop Size [m]"]["mean"] == pytest.approx(d_h)
    assert r["Hinze Drop Size / Voxel Size [-]"]["mean"] == pytest.approx(d_h / 1e-4)


# ---------------------------------------------------------------- (b) VOF structures

RADII = [0.01, 0.015, 0.02, 0.025]
SLAB = (0.2, 0.2, 0.05)
VOXEL = 1e-3
OIL_R = 0.012


def vof_surface(pad_degenerate: bool = True):
    parts = [pv.Sphere(radius=r, center=(0.1 * i, 0.5, 0.0),
                       theta_resolution=80, phi_resolution=80)
             for i, r in enumerate(RADII)]
    parts.append(pv.Box(bounds=(0, SLAB[0], 0, SLAB[1], 0, SLAB[2])).triangulate())
    parts.append(pv.Sphere(radius=1e-3, center=(0.5, 0.5, 0.5)))  # 4.2 voxels: debris
    parts.append(pv.Plane(center=(0.5, 0.0, 0.5), i_size=0.1, j_size=0.1).triangulate())
    # inward normals = fluid 2 inside (M-Star orients normals toward fluid 2)
    parts.append(pv.Sphere(radius=OIL_R, center=(0.5, 0.3, 0.3), theta_resolution=80,
                           phi_resolution=80).flip_faces())
    surf = pv.merge(parts)
    if pad_degenerate:  # M-Star pads its VOF surface output with (0, 0, 0) triangles
        faces = np.r_[surf.faces, np.tile([3, 0, 0, 0], 50)]
        surf = pv.PolyData(surf.points, faces)
    return surf


@pytest.fixture
def vof_case(tmp_path):
    case = tmp_path / "case"
    out = case / "out" / "Output"
    for t in (1.0, 2.0):
        surf = vof_surface()
        surf.field_data["TIME"] = np.array([t])
        d = out / "VolumeVOFSurface" / "block0"
        d.mkdir(parents=True, exist_ok=True)
        surf.save(d / f"VOFSurface.{t:.5e}.vtp")
    # slice VOF output must be ignored
    sl = out / "SliceVOFSurface" / "block0"
    sl.mkdir(parents=True)
    pv.Sphere(radius=0.05).save(sl / "VOFSurface.1.00000e+00.vtp")
    # M-Star's VOF structure point cloud must be ignored
    vs = out / "VolumeVOFStructures" / "block0"
    vs.mkdir(parents=True)
    pv.PolyData(np.zeros((3, 3))).save(vs / "vof_structures.1.00000e+00.vtp")
    return case


def fluids(v1=FLUID1_VOL, v2=FLUID2_VOL):
    t = np.array([0.0, 10.0])
    return [(1, "Water", t, np.full(2, v1)), (2, "Oil", t, np.full(2, v2))]


def expected_vof():
    vs = np.array([4 / 3 * np.pi * r ** 3 for r in RADII])
    as_ = np.array([4 * np.pi * r ** 2 for r in RADII])
    v_slab = np.prod(SLAB)
    a_slab = 2 * (SLAB[0] * SLAB[1] + SLAB[0] * SLAB[2] + SLAB[1] * SLAB[2])
    v_all, a_all = np.r_[vs, v_slab], np.r_[as_, a_slab]
    d = (6 * v_all / np.pi) ** (1 / 3)
    d32 = 6 * v_all.sum() / a_all.sum()
    v_oil = 4 / 3 * np.pi * OIL_R ** 3
    return {
        "n_drops_fluid1": len(RADII) + 1,
        "d32_m_fluid1": d32,
        "d32_dx_fluid1": d32 / VOXEL,
        "d10_m_fluid1": d.mean(),
        "d43_m_fluid1": (d ** 4).sum() / (d ** 3).sum(),
        "d_max_m_fluid1": d.max(),
        "F_disp_fluid1": v_all.sum() / FLUID1_VOL,
        "n_drops_fluid2": 1,
        "d32_m_fluid2": 2 * OIL_R,
        "F_disp_fluid2": v_oil / FLUID2_VOL,
        "A_drops_m2": a_all.sum() + 4 * np.pi * OIL_R ** 2,
    }


def test_analyze_vof_surface_regions():
    s = mss.analyze_vof_surface(pv, vof_surface())
    assert s["source"] == "connectivity"
    assert len(s["vol"]) == len(RADII) + 4
    closed = s["closed"]
    assert (~closed).sum() == 1  # the open plane
    assert np.max(s["vol"][closed]) == pytest.approx(np.prod(SLAB))
    # orientation fallback: outward normals = fluid 1 inside, flipped sphere = fluid 2
    inside = s["inside"][closed]
    assert (inside == 1).sum() == len(RADII) + 2 and (inside == 0).sum() == 1
    assert np.isnan(s["inside"][~closed]).all()


def test_analyze_vof_surface_merges_seams():
    sphere = pv.Sphere(radius=0.01, theta_resolution=40, phi_resolution=40)
    upper = sphere.clip("z", value=0.0, invert=False).triangulate()
    lower = sphere.clip("z", value=0.0).triangulate()
    upper.points[:, :] += 1e-11  # seam vertices off by 1e-8 voxels
    s = mss.analyze_vof_surface(pv, pv.merge([upper, lower], merge_points=False),
                                voxel_size=VOXEL)
    assert len(s["vol"]) == 1 and s["closed"][0]


def test_vof_structures_metrics(vof_case):
    a_int = (np.array([0.0, 10.0]), np.full(2, 2.0))
    rows, series = mss.vof_structures(vof_case, min_voxels=8, voxel_size=VOXEL,
                                      t_start=0.5, fluids=fluids(), a_int=a_int)
    exp = expected_vof()
    np.testing.assert_allclose(series["t"], [1.0, 2.0])  # VOF structures cloud ignored
    for key, value in exp.items():
        np.testing.assert_allclose(series[key], value, rtol=5e-3, err_msg=key)
    np.testing.assert_allclose(series["coverage"], exp["A_drops_m2"] / 2.0, rtol=5e-3)

    r = by_variable(rows)
    assert r["Number of Drops n_drops (Fluid 1 drops) [-]"]["mean"] == len(RADII) + 1
    assert r["Number of Drops n_drops (Fluid 2 drops) [-]"]["mean"] == 1
    assert r["Dispersed Fraction F_disp (Fluid 2 in drops / Fluid 2 volume) [-]"]["mean"] \
        == pytest.approx(exp["F_disp_fluid2"], rel=5e-3)
    assert r["Sauter Diameter d32 / Voxel Size (VOF, Fluid 1 drops) [-]"]["mean"] \
        == pytest.approx(exp["d32_dx_fluid1"], rel=5e-3)
    assert r["Drop Interface Coverage A_drops/A_int [-]"]["steady_method"] == "vof"

    with open(vof_case / "vof_structures.csv") as fh:
        ts = list(csv.DictReader(fh))
    assert len(ts) == 2
    assert ts[0]["n_open"] == "1" and ts[0]["n_debris"] == "1"
    with open(vof_case / "vof_dsd.csv") as fh:
        dsd = list(csv.DictReader(fh))
    assert len(dsd) == 2 * (len(RADII) + 2)
    assert {d["phase"] for d in dsd} == {"Fluid 1", "Fluid 2"}


def test_vof_structures_min_voxel_filter(vof_case):
    _, series = mss.vof_structures(vof_case, 0, VOXEL, 0.0, fluids())
    assert series["n_drops_fluid1"][0] == len(RADII) + 2  # debris sphere kept
    # threshold above the smallest real drop (r = 1 cm, ~4189 voxels) removes it too
    _, series = mss.vof_structures(vof_case, 5000, VOXEL, 0.0, fluids())
    assert series["n_drops_fluid1"][0] == len(RADII)


def test_vof_structures_phase_from_vti(tmp_path, capsys):
    out = tmp_path / "case" / "Output"
    surf = pv.Sphere(radius=0.01, theta_resolution=60, phi_resolution=60)  # outward
    (out / "VolumeVOFSurface" / "block0").mkdir(parents=True)
    surf.save(out / "VolumeVOFSurface" / "block0" / "VOFSurface.1.00000e+00.vtp")
    img = pv.ImageData(dimensions=(31, 31, 31), spacing=(VOXEL,) * 3,
                       origin=(-0.015,) * 3)
    r = np.linalg.norm(img.cell_centers().points, axis=1)
    img.cell_data["Fluid 1 (Water) Volume Fraction (-)"] = (r > 0.01).astype(float)
    img.cell_data["Fluid 2 (Oil) Volume Fraction (-)"] = (r <= 0.01).astype(float)
    (out / "Volume" / "block0").mkdir(parents=True)
    img.save(out / "Volume" / "block0" / "Volume.1.00000e+00.vti")

    _, series = mss.vof_structures(tmp_path / "case", 8, VOXEL, 0.0, fluids())
    assert series["n_drops_fluid1"][0] == 0 and series["n_drops_fluid2"][0] == 1
    err = capsys.readouterr().err
    assert "disagrees with triangle orientation for 1 of 1" in err


def test_vof_structures_structure_id_array(tmp_path):
    surf = pv.merge([pv.Sphere(radius=0.01, center=(0, 0, 0),
                               theta_resolution=80, phi_resolution=80),
                     pv.Sphere(radius=0.01, center=(0.1, 0, 0),
                               theta_resolution=80, phi_resolution=80)])
    # one ID for both spheres: treated as a single structure
    surf.cell_data["VOF Structure ID"] = np.zeros(surf.n_cells, int)
    s = mss.analyze_vof_surface(pv, surf)
    assert "VOF Structure ID" in s["source"]
    assert len(s["vol"]) == 1 and s["closed"][0]
    assert s["vol"][0] == pytest.approx(2 * 4 / 3 * np.pi * 0.01 ** 3, rel=5e-3)


# ---------------------------------------------------------------- (c) phase uniformity

def write_vti(path: Path, t: float, with_mask: bool = True) -> None:
    img = pv.ImageData(dimensions=(21, 41, 21), spacing=(0.01, 0.01, 0.01),
                       origin=(-0.1, 0.0, -0.1))
    cc = img.cell_centers().points
    img.cell_data["Volume Fraction"] = cc[:, 1] / 0.4  # linear in y: 0..1
    if with_mask:
        r = np.hypot(cc[:, 0], cc[:, 2])
        img.cell_data["Solid Flag"] = (r > 0.09).astype(np.uint8)
    img.field_data["TIME"] = np.array([t])
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def test_phase_uniformity_vertical_gradient(tmp_path):
    case = tmp_path / "case"
    for t in (1.0, 2.0):
        write_vti(case / "Output" / "Volume" / "block0" / f"Volume.{t:.5e}.vti", t)
    rows, series = mss.phase_uniformity(case, 10, 3, t_start=0.0, vertical_axis="y")

    # zone means are the bin centers (b + 0.5)/10, all bins equal volume
    centers = (np.arange(10) + 0.5) / 10
    expected_cov = centers.std() / centers.mean()
    np.testing.assert_allclose(series["cov"], expected_cov, rtol=1e-6)
    assert rows[0]["variable"] == "Zonal CoV of Phase Fraction [-]"
    assert rows[0]["steady_method"] == "zonal"
    assert rows[0]["mean"] == pytest.approx(expected_cov, rel=1e-6)

    prof = [r for r in series["profile"] if r["time_s"] == 1.0]
    assert len(prof) == 40
    for r in prof:
        assert r["phi_mean"] == pytest.approx(r["z_m"] / 0.4, rel=1e-5)
    assert (case / "phase_uniformity.csv").exists()
    assert (case / "holdup_profile.csv").exists()


def test_phase_uniformity_two_fluids_and_solid(tmp_path):
    img = pv.ImageData(dimensions=(21, 41, 21), spacing=(0.01, 0.01, 0.01),
                       origin=(-0.1, 0.0, -0.1))
    cc = img.cell_centers().points
    phi1 = cc[:, 1] / 0.4
    solid = np.hypot(cc[:, 0], cc[:, 2]) > 0.09
    img.cell_data["Fluid 1 (Water) Volume Fraction (-)"] = np.where(solid, 0.0, phi1)
    img.cell_data["Fluid 2 (Oil) Volume Fraction (-)"] = np.where(solid, 0.0, 1 - phi1)
    img.cell_data["VOF Structure ID (-)"] = np.zeros(img.n_cells)
    img.field_data["TIME"] = np.array([1.0])
    case = tmp_path / "case"
    (case / "Output" / "Volume" / "block0").mkdir(parents=True)
    img.save(case / "Output" / "Volume" / "block0" / "Volume.1.00000e+00.vti")
    bc = pv.ImageData(dimensions=(3, 3, 3))
    bc.cell_data["Boundary Condition"] = np.zeros(bc.n_cells)
    (case / "Output" / "BoundaryConditions" / "block0").mkdir(parents=True)
    bc.save(case / "Output" / "BoundaryConditions" / "block0" /
            "BoundaryConditions.0.00000e+00.vti")

    rows, series = mss.phase_uniformity(case, 10, 3, t_start=0.0)
    centers = (np.arange(10) + 0.5) / 10
    r = by_variable(rows)
    assert r["Zonal CoV of Phase Fraction (Fluid 1) [-]"]["mean"] == pytest.approx(
        centers.std() / centers.mean(), rel=1e-6)
    assert r["Zonal CoV of Phase Fraction (Fluid 2) [-]"]["mean"] == pytest.approx(
        centers.std() / (1 - centers).mean(), rel=1e-6)
    prof = series["profile"][0]
    assert prof["phi_mean_fluid1"] + prof["phi_mean_fluid2"] == pytest.approx(1.0)


def test_phase_uniformity_missing_array(tmp_path, capsys):
    img = pv.ImageData(dimensions=(3, 3, 3))
    img.cell_data["Pressure (Pa)"] = np.zeros(img.n_cells)
    img.save(tmp_path / "Volume.1.00000e+00.vti")
    rows, series = mss.phase_uniformity(tmp_path, 10, 3, 0.0)
    assert rows == [] and series == {}
    assert "Pressure (Pa)" in capsys.readouterr().err


# ---------------------------------------------------------------- end to end

def test_main_dispersion_end_to_end(stats_case, vof_case, tmp_path):
    case, _ = stats_case
    for p in (vof_case / "out").rglob("*.vtp"):
        dest = case / p.relative_to(vof_case)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    write_vti(case / "Output" / "Volume" / "block0" / "Volume.1.00000e+00.vti", 1.0)

    mss.main([str(case), "--time", "0.5", "--dispersion", "--dispersed-fraction", "0.2",
              "--voxel-size", str(VOXEL), "--sigma", "0.05", "--rho-continuous", "1000",
              "--uniformity", "--plot", "mean velocity"])

    with open(case / "stats_summary.csv") as fh:
        methods = {r["steady_method"] for r in csv.DictReader(fh)}
    assert {"interface", "vof", "zonal", "hinze"} <= methods
    assert (case / "stats_plots" / "dispersion.html").exists()
    with open(case / "stats_summary_means.csv") as fh:
        cols = next(csv.reader(fh))
    for q in ("interfacial area", "d32", "dispersed fraction", "n_drops", "cov",
              "coverage", "hinze drop size / voxel", "d32 / voxel size"):
        assert any(q in c.lower() for c in cols), q


def test_default_run_has_no_dispersion_outputs(stats_case):
    case, _ = stats_case
    mss.main([str(case), "--time", "0.5"])
    with open(case / "stats_summary.csv") as fh:
        methods = {r["steady_method"] for r in csv.DictReader(fh)}
    assert methods == {"user"}
    assert not (case / "vof_structures.csv").exists()
