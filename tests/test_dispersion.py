"""Tests for the phase-dispersion metrics of mstar_stats_summary.py (synthetic data)."""

import csv
import os
import shutil
import subprocess
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
    assert r["Interfacial Area A_int [m^2]"]["mean"] == pytest.approx(3e-3)
    assert r["Plateau Interfacial Area A_ss = a_ss V_liq [m^2]"]["mean"] == pytest.approx(3e-3, rel=0.01)
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
    assert rows[0]["variable"] == "Specific Interfacial Area [1/m]"
    assert not any("d32" in r["variable"] for r in rows)
    assert rows[0]["mean"] == pytest.approx(1.5)
    assert "d32" not in series
    assert "--dispersed-fraction not given" in capsys.readouterr().err


def test_dispersion_kinetics_exponential():
    a0, a_ss, tau = 10.0, 100.0, 1.5
    t = np.arange(0.0, 30.0, 0.05)
    a = a_ss - (a_ss - a0) * np.exp(-t / tau)
    k = mss.dispersion_kinetics(t, a, window=1.0, tol=0.005)
    assert k["a0"] == pytest.approx(a0)
    assert k["a_ss"] == pytest.approx(a_ss, rel=0.01)
    assert k["E"] == pytest.approx(a_ss / a0, rel=0.01)
    assert k["t90"] == pytest.approx(tau * np.log(10), rel=0.03)
    assert k["t95"] == pytest.approx(-tau * np.log(0.05), rel=0.03)
    assert k["tau"] == pytest.approx(tau, rel=0.05)
    assert k["plateau_cov"] < 0.01
    T = t[-1]
    assert k["int_a"] == pytest.approx(a_ss * T - (a_ss - a0) * tau * (1 - np.exp(-T / tau)),
                                       rel=1e-3)


def test_dispersion_kinetics_no_plateau_warns(tmp_path, capsys):
    stats = tmp_path / "Stats"
    t = np.arange(0.0, 5.0, 0.05)
    write_table(stats / "Interface.txt", ["Time [s]", "Area [m^2]"], [t, 1e-3 * (1 + t)])
    rows, series = mss.interface_metrics(mss.find_stats_files(stats), stats, 0.0, None,
                                         1e-3, vessel_diameter=0.1)
    assert np.isnan(series["kinetics"]["t_ss_a"])
    # kinetics still reported, from the last-window mean of a
    r = by_variable(rows)
    e = r["Interface Enhancement E = a_ss/a0 [(1/m)/(1/m)]"]
    assert e["steady_method"] == "a(t) kinetics (no plateau, last window)"
    assert np.isfinite(e["mean"]) and e["mean"] > 1
    assert np.isfinite(r["Dispersion Time t95 [s]"]["mean"])
    assert "Interfacial Area Plateau Start t_ss_a [s]" not in r
    err = capsys.readouterr().err
    assert "dispersion still developing" in err
    assert "flat-interface estimate" in err  # a0 = 1 vs pi*0.01/4/1e-3 = 7.85 1/m


def test_series_row_falls_back_to_last_samples(capsys):
    t = np.arange(0.0, 4.0, 1.0)
    row = mss.series_row("f.csv", "X [-]", t, t, 10.0, "vof")
    assert row["steady_method"] == "vof (fallback, before steady start)"
    assert row["steady_start_s"] == pytest.approx(2.25)
    assert row["mean"] == pytest.approx(3.0)
    assert "averaging the last samples" in capsys.readouterr().err
    assert mss.series_row("f.csv", "X [-]", t, np.full(4, np.nan), 0.0, "vof") is None


def test_no_velocity_plateau_still_processed(tmp_path, capsys):
    stats = tmp_path / "case" / "Stats"
    t = np.arange(0.0, 8.0001, 0.1)
    write_table(stats / "Fluid.txt", ["Time [s]", "Mean Velocity [m/s]", "Fluid Volume [m^3]"],
                [t, 0.1 * t, np.full_like(t, 1e-3)])
    write_table(stats / "Interface.txt", ["Time [s]", "Area [m^2]"], [t, 1e-3 * (1 + t)])
    mss.main([str(tmp_path / "case"), "--dispersion", "--voxel-size", "1e-3"])
    assert "no mean-velocity plateau" in capsys.readouterr().err
    with open(tmp_path / "case" / "stats_summary.csv") as fh:
        rows = by_variable(list(csv.DictReader(fh)))
    vel = rows["Mean Velocity [m/s]"]
    assert vel["steady_method"] == mss.FALLBACK
    assert float(vel["steady_start_s"]) == pytest.approx(6.0)  # last 25% of 8 s
    assert rows["Specific Interfacial Area [1/m]"]["steady_method"] \
        == f"interface ({mss.FALLBACK})"
    assert "no plateau" in rows["Interface Enhancement E = a_ss/a0 [(1/m)/(1/m)]"]["steady_method"]
    with open(tmp_path / "case" / "dispersion_timeseries.csv") as fh:
        ts = list(csv.DictReader(fh))
    assert len(ts) == t.size
    assert float(ts[0]["E = a/a0 [(1/m)/(1/m)]"]) == pytest.approx(1.0)
    assert float(ts[-1]["E = a/a0 [(1/m)/(1/m)]"]) == pytest.approx(9.0)
    assert float(ts[-1]["int_a [s/m]"]) == pytest.approx(8 + 8 ** 2 / 2, rel=1e-6)


def test_interface_area_column_selection(tmp_path, capsys):
    stats = tmp_path / "Stats"
    t = np.arange(0.0, 1.0, 0.1)
    write_table(stats / "Interface.txt", ["Time [s]", "Wall Area [m^2]", "Area [m^2]"],
                [t, np.ones_like(t), np.full_like(t, 2.0)])
    files = mss.find_stats_files(stats)
    rows, _ = mss.interface_metrics(files, stats, 0.0, None, 1.0)
    assert rows[0]["mean"] == pytest.approx(1.0)
    assert "several area columns" in capsys.readouterr().err
    rows, _ = mss.interface_metrics(files, stats, 0.0, None, 1.0, area_column="wall")
    assert rows[0]["mean"] == pytest.approx(1.0)


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
    assert r["Hinze Drop Size / Voxel Size [m/m]"]["mean"] == pytest.approx(d_h / 1e-4)

    rows = mss.resolution_check(mss.find_stats_files(stats), 1.0, 1e-4, sigma=0.05,
                                rho_c=1000.0, k_phi=3.0, eps_ratio=10.0, phi=0.5)
    d_c = 0.725 * 2.5 * (0.05 / 1000.0) ** 0.6 * 10.0 ** -0.4
    assert by_variable(rows)["Hinze Max Stable Drop Size [m]"]["mean"] == pytest.approx(d_c)


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
    assert r["Number of Drops n_drops (Fluid 1 drops) [count]"]["mean"] == len(RADII) + 1
    assert r["Number of Drops n_drops (Fluid 2 drops) [count]"]["mean"] == 1
    assert r["Dispersed Fraction F_disp (Fluid 2 in drops / Fluid 2 volume) [m^3/m^3]"]["mean"] \
        == pytest.approx(exp["F_disp_fluid2"], rel=5e-3)
    assert r["Sauter Diameter d32 / Voxel Size (VOF, Fluid 1 drops) [m/m]"]["mean"] \
        == pytest.approx(exp["d32_dx_fluid1"], rel=5e-3)
    assert r["Drop Interface Coverage A_drops/A_int [m^2/m^2]"]["steady_method"] == "vof"

    with open(vof_case / "vof_structures.csv") as fh:
        ts = list(csv.DictReader(fh))
    assert len(ts) == 2
    assert ts[0]["n_open [count]"] == "1" and ts[0]["n_debris [count]"] == "1"
    with open(vof_case / "vof_dsd.csv") as fh:
        dsd = list(csv.DictReader(fh))
    assert len(dsd) == 2 * (len(RADII) + 2)
    assert {d["phase"] for d in dsd} == {"Fluid 1", "Fluid 2"}


def test_vof_structures_min_voxel_filter(vof_case):
    _, series = mss.vof_structures(vof_case, 0, VOXEL, 0.0, fluids(), min_d_dx=0)
    assert series["n_drops_fluid1"][0] == len(RADII) + 2  # debris sphere kept
    # default min_d_dx = 4 excludes the 2-voxel sphere
    _, series = mss.vof_structures(vof_case, 0, VOXEL, 0.0, fluids())
    assert series["n_drops_fluid1"][0] == len(RADII) + 1
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


def test_vof_structures_two_block_seam(tmp_path):
    out = tmp_path / "case" / "Output"
    sphere = pv.Sphere(radius=0.01, theta_resolution=40, phi_resolution=40)  # outward
    halves = [sphere.clip("z", value=0.0).triangulate(),
              sphere.clip("z", value=0.0, invert=False).triangulate()]
    halves[1].points[:, :] += 1e-11  # seam vertices do not coincide exactly
    for b, (half, z0) in enumerate(zip(halves, (-0.015, 0.0))):
        d = out / "VolumeVOFSurface" / f"block{b}"
        d.mkdir(parents=True)
        pv.PolyData(half.points, half.faces).save(d / "VOFSurface.1.00000e+00.vtp")
        img = pv.ImageData(dimensions=(31, 31, 16), spacing=(VOXEL,) * 3,
                           origin=(-0.015, -0.015, z0))
        r = np.linalg.norm(img.cell_centers().points, axis=1)
        img.cell_data["Fluid 1 (Water) Volume Fraction (-)"] = (r > 0.01).astype(float)
        img.cell_data["Fluid 2 (Oil) Volume Fraction (-)"] = (r <= 0.01).astype(float)
        (out / "Volume" / f"block{b}").mkdir(parents=True)
        img.save(out / "Volume" / f"block{b}" / "Volume.1.00000e+00.vti")

    _, series = mss.vof_structures(tmp_path / "case", 8, VOXEL, 0.0, fluids())
    assert series["n_open"][0] == 0
    assert series["n_drops_fluid1"][0] == 0 and series["n_drops_fluid2"][0] == 1
    assert series["d32_m_fluid2"][0] == pytest.approx(0.02, rel=0.01)


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
    np.testing.assert_allclose(series["cov_10x3"], expected_cov, rtol=1e-6)
    assert rows[0]["variable"] == "Zonal CoV of Phase Fraction (10x3) [m^3/m^3]"
    assert rows[0]["steady_method"] == "zonal"
    assert rows[0]["mean"] == pytest.approx(expected_cov, rel=1e-6)
    assert not any("Segregation" in r["variable"] for r in rows)  # single array

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
    assert r["Zonal CoV of Phase Fraction (Fluid 1, 10x3) [m^3/m^3]"]["mean"] == pytest.approx(
        centers.std() / centers.mean(), rel=1e-6)
    assert r["Zonal CoV of Phase Fraction (Fluid 2, 10x3) [m^3/m^3]"]["mean"] == pytest.approx(
        centers.std() / (1 - centers).mean(), rel=1e-6)
    prof = series["profile"][0]
    assert prof["phi_mean_fluid1"] + prof["phi_mean_fluid2"] == pytest.approx(1.0)


def write_two_fluid_vti(path: Path, t: float, phi1: np.ndarray | None = None,
                        func=None) -> None:
    img = pv.ImageData(dimensions=(21, 41, 21), spacing=(0.01, 0.01, 0.01),
                       origin=(-0.1, 0.0, -0.1))
    cc = img.cell_centers().points
    p1 = func(cc) if func else phi1
    img.cell_data["Fluid 1 (Water) Volume Fraction (-)"] = p1
    img.cell_data["Fluid 2 (Oil) Volume Fraction (-)"] = 1.0 - p1
    img.field_data["TIME"] = np.array([t])
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


@pytest.mark.parametrize("name, func, s_expected, continuous", [
    ("stratified", lambda c: (c[:, 1] < 0.2).astype(float), 1.0, "both"),
    ("uniform", lambda c: np.full(len(c), 0.5), 0.0, "none"),
    # isolated 1-cell oil drops on a 4-cell lattice in water
    ("drops", lambda c: 1.0 - ((np.rint((c[:, 0] + 0.1) / 0.01 - 0.5) % 4 == 0)
                               & (np.rint(c[:, 1] / 0.01 - 0.5) % 4 == 0)
                               & (np.rint((c[:, 2] + 0.1) / 0.01 - 0.5) % 4 == 0)),
     None, "Fluid 1 (Water)"),
])
def test_segregation_index_and_continuity(tmp_path, name, func, s_expected, continuous):
    case = tmp_path / name
    for t in (1.0, 2.0, 3.0):
        write_two_fluid_vti(case / "Output" / "Volume" / "block0" / f"Volume.{t:.5e}.vti",
                            t, func=func)
    rows, series = mss.phase_uniformity(case, 10, 3, t_start=0.0,
                                        zone_scales=[(5, 2), (10, 3)])
    r = by_variable(rows)
    if s_expected is not None:
        s = r["Segregation Index S (Fluid 1, 10x3) [m^3/m^3]"]["mean"]
        assert s == pytest.approx(s_expected, abs=1e-9)
    assert "Segregation Index S (Fluid 2, 5x2) [m^3/m^3]" in r
    assert r["Continuous Phase"]["mean"].startswith(continuous)
    with open(case / "phase_uniformity.csv") as fh:
        cols = next(csv.reader(fh))
    assert {"S_fluid1_5x2 [m^3/m^3]", "S_fluid2_10x3 [m^3/m^3]", "cov_fluid1_10x3 [m^3/m^3]",
            "time [s]", "phi_mean_fluid1 [m^3/m^3]", "n_zones_10x3 [count]"} <= set(cols)
    assert (case / "phase_continuity.csv").exists()


def test_homogeneity_time(tmp_path):
    # stratified layer relaxes to uniform: phi1 = 0.5 + 0.5*sign(0.2 - y)*exp(-t)
    case = tmp_path / "case"
    for t in np.arange(0.0, 15.0, 0.5):
        write_two_fluid_vti(
            case / "Output" / "Volume" / "block0" / f"Volume.{t:.5e}.vti", t,
            func=lambda c, t=t: 0.5 + 0.5 * np.sign(0.2 - c[:, 1]) * np.exp(-t))
    rows, _ = mss.phase_uniformity(case, 10, 3, t_start=0.0, window=1.0, tol=0.02)
    t_s = by_variable(rows)["Homogeneity Time t_S (Fluid 1, 10x3) [s]"]["mean"]
    assert t_s == pytest.approx(-np.log(0.05), abs=0.5)  # S(t) = exp(-t)


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
    write_two_fluid_vti(case / "Output" / "Volume" / "block0" / "Volume.1.00000e+00.vti",
                        1.0, func=lambda c: (c[:, 1] < 0.2).astype(float))

    mss.main([str(case), "--time", "0.5", "--dispersion", "--dispersed-fraction", "0.2",
              "--voxel-size", str(VOXEL), "--sigma", "0.05", "--rho-continuous", "1000",
              "--uniformity", "--disp-time", "0.5", "--zone-scales", "5x2", "10x3",
              "--plot-html", "--plot", "mean velocity"])

    with open(case / "stats_summary.csv") as fh:
        rows = list(csv.DictReader(fh))
    methods = {r["steady_method"] for r in rows}
    assert {"interface", "a(t) kinetics", "user"} <= methods
    seg = [r for r in rows if r["variable"].startswith("Segregation Index")]
    assert seg and all(r["steady_method"] == "user" and float(r["steady_start_s"]) == 0.5
                       for r in seg)
    assert (case / "stats_plots" / "dispersion.html").exists()
    with open(case / "dispersion_timeseries.csv") as fh:
        ts = list(csv.DictReader(fh))
    assert set(ts[0]) >= {"time [s]", "A_int [m^2]", "V_liq [m^3]", "a [1/m]",
                          "E = a/a0 [(1/m)/(1/m)]", "a_norm = (a - a0)/(a_ss - a0) [(1/m)/(1/m)]",
                          "int_a [s/m]", "eps [W/kg]", "d_hinze [m]", "d_hinze/dx [m/m]"}
    assert float(ts[0]["E = a/a0 [(1/m)/(1/m)]"]) == pytest.approx(1.0)
    assert float(ts[-1]["E = a/a0 [(1/m)/(1/m)]"]) == pytest.approx(3.0)
    assert float(ts[-1]["a_norm = (a - a0)/(a_ss - a0) [(1/m)/(1/m)]"]) == pytest.approx(1.0)
    d_h = 0.725 * (0.05 / 1000.0) ** 0.6 * 1.0 ** -0.4
    assert float(ts[-1]["d_hinze [m]"]) == pytest.approx(d_h)
    assert float(ts[-1]["d_hinze/dx [m/m]"]) == pytest.approx(d_h / VOXEL)
    with open(case / "stats_summary_means.csv") as fh:
        cols = next(csv.reader(fh))
    for q in ("interfacial area", "d32", "dispersed fraction", "n_drops", "cov",
              "coverage", "hinze drop size / voxel", "d32 / voxel size",
              "interface enhancement", "dispersion time t95",
              "plateau cov", "time-integrated", "segregation index s (fluid 1, 5x2)",
              "segregation index s (fluid 1, 10x3)", "continuous phase"):
        assert any(q in c.lower() for c in cols), q

    # without --disp-time the VTK metrics start at the a(t) plateau (step at t = 1 s)
    mss.main([str(case), "--time", "0.5", "--dispersion", "--voxel-size", str(VOXEL),
              "--uniformity"])
    with open(case / "stats_summary.csv") as fh:
        rows = list(csv.DictReader(fh))
    t_ss = float(next(r["mean"] for r in rows
                      if r["variable"].startswith("Interfacial Area Plateau Start")))
    vof = [r for r in rows if r["file"] == "vof_structures.csv"]
    assert vof and all(r["steady_method"] == "a(t) plateau"
                       and float(r["steady_start_s"]) == pytest.approx(t_ss) for r in vof)


def test_batch_ignores_hidden_folders(stats_case, tmp_path):
    case, stats = stats_case
    (case / ".archive" / "Stats").mkdir(parents=True)
    (case / ".archive" / "Stats" / "Fluid.txt").write_text((stats / "Fluid.txt").read_text())
    hidden = tmp_path / ".old_case" / "Stats"
    hidden.mkdir(parents=True)
    (hidden / "Fluid.txt").write_text((stats / "Fluid.txt").read_text())
    cases = mss.find_batch_cases(tmp_path)
    assert [c.name for c, _ in cases] == ["case"]
    assert cases[0][1] == stats
    assert all(".archive" not in str(f) for f in mss.find_stats_files(case))


def test_batch_discovers_cases_nested_under_reactors(stats_case, tmp_path):
    _, stats = stats_case
    root = tmp_path / "step8"
    expected = []
    for reactor, subcases in (("10L", ("run-a", "run-b")),
                              ("R101", ("run-c",)),
                              ("R901", ("run-d", "run-e"))):
        for subcase in subcases:
            case = root / reactor / subcase
            case_stats = case / "Stats"
            case_stats.mkdir(parents=True)
            (case_stats / "Fluid.txt").write_text((stats / "Fluid.txt").read_text())
            expected.append((case, case_stats))

    cases = mss.find_batch_cases(root)
    assert cases == sorted(expected)


def test_batch_mstar_out_layout(stats_case, tmp_path):
    _, stats = stats_case
    root = tmp_path / "step8"
    for rel in ("CHP_10L/case-a", "CHP_10L/case-b", "CHP_R101/case-c"):
        (root / rel / "out" / "Stats").mkdir(parents=True)
        (root / rel / "input.xml").write_text("<input/>")
    (root / "loose" / "out" / "Stats").mkdir(parents=True)  # no input.xml
    ids = [c.relative_to(root).as_posix() for c, _ in mss.find_batch_cases(root)]
    assert ids == ["CHP_10L/case-a", "CHP_10L/case-b", "CHP_R101/case-c", "loose"]


def test_batch_output_root_and_case_filter(stats_case, tmp_path):
    _, stats = stats_case
    root = tmp_path / "step8"
    for rel in ("10L/run-a", "10L/run-b", "R101/run-a"):
        (root / rel / "Stats").mkdir(parents=True)
        (root / rel / "Stats" / "Fluid.txt").write_text((stats / "Fluid.txt").read_text())
    before = sorted(p for p in root.rglob("*"))
    proj = tmp_path / "proj"
    mss.main([str(root), "--batch", "--time", "0.5", "--plot", "velocity",
              "--output-root", str(proj), "--cases", "10L/run-a", "R101/run-a"])
    assert sorted(p for p in root.rglob("*")) == before
    assert (proj / "10L" / "run-a" / "stats_summary.csv").is_file()
    # no HTML by default: the app plots from the CSVs
    assert not (proj / "R101" / "run-a" / "stats_plots").exists()
    assert not list(proj.rglob("*.html"))
    assert not (proj / "10L" / "run-b").exists()
    with open(proj / "batch_summary_means.csv") as fh:
        assert [r["case"] for r in csv.DictReader(fh)] == ["10L/run-a", "R101/run-a"]
    with open(proj / "batch_summary_settings.csv") as fh:
        assert [r["case"] for r in csv.DictReader(fh)] == ["10L/run-a", "R101/run-a"]


def test_ts_labels_have_units():
    for key, label in (
            ("time_s", "time [s]"), ("E", "E = a/a0 [(1/m)/(1/m)]"),
            ("A_drops_m2", "A_drops [m^2]"), ("V_drops_m3_fluid2", "V_drops_fluid2 [m^3]"),
            ("d32_m_fluid1", "d32_fluid1 [m]"), ("d32_dx_fluid1", "d32/dx_fluid1 [m/m]"),
            ("d_max_m_fluid1", "d_max_fluid1 [m]"), ("n_drops_fluid1", "n_drops_fluid1 [count]"),
            ("S_fluid1_10x3", "S_fluid1_10x3 [m^3/m^3]"), ("cov_10x3", "cov_10x3 [m^3/m^3]"),
            ("share_fluid2", "share_fluid2 [m^3/m^3]"), ("z_m", "z [m]"),
            ("continuous_phase", "continuous_phase")):
        assert mss.ts_label(key) == label


def test_migrate_units(tmp_path):
    run = tmp_path / "run"
    (run / "c1").mkdir(parents=True)
    (run / "c1" / "dispersion_timeseries.csv").write_text("time_s,E,a_1_m\n0,1,2\n")
    (run / "c1" / "stats_summary.csv").write_text(
        "file,variable,unit,mean\nx,Interface Enhancement E = a_ss/a0 [-],-,2.5\n"
        "x,Power Number [-],-,4\n")
    (run / "batch_summary_means.csv").write_text(
        "case,\"Segregation Index S (Fluid 1, 10x3) [-]\",Mean Velocity [m/s]\nc1,0.2,0.1\n")
    (run / "batch_summary_means.html").write_text("old")
    assert mss.migrate_units(run) == 3
    assert (run / "c1" / "dispersion_timeseries.csv").read_text().splitlines()[0] \
        == "time [s],E = a/a0 [(1/m)/(1/m)],a [1/m]"
    with open(run / "c1" / "stats_summary.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["variable"] == "Interface Enhancement E = a_ss/a0 [(1/m)/(1/m)]"
    assert rows[0]["unit"] == "(1/m)/(1/m)"
    assert rows[1]["variable"] == "Power Number [W/W]" and rows[1]["unit"] == "W/W"
    assert next(csv.reader(open(run / "batch_summary_means.csv")))[1] \
        == "Segregation Index S (Fluid 1, 10x3) [m^3/m^3]"
    html = (run / "batch_summary_means.html").read_text()
    assert "Segregation Index S" in html and "[-]" not in html
    assert mss.migrate_units(run) == 0  # idempotent


def test_case_settings_dimensionless_groups(stats_case):
    case, stats = stats_case
    (case / "input.xml").write_text(
        "<input><system><dx>0.001</dx><dt>1e-4</dt></system>"
        "<impeller><name>agitator</name><freq>120</freq><diameter>0.1</diameter></impeller>"
        "<fluidModel><fluid1Name>Water</fluid1Name><fluid2Name>Oil</fluid2Name>"
        "<rheology><density>1000</density><nu>1e-6</nu></rheology>"
        "<rheology><density>800</density><nu>2e-6</nu></rheology>"
        "<surfaceTension>0.02</surfaceTension></fluidModel></input>")
    s = mss.case_settings(case, stats)
    n, d = 2.0, 0.1
    assert s["Impeller diameter (agitator) [m]"] == pytest.approx(d)
    assert s["Continuous phase (largest initial volume)"] == "Fluid 1 (Water)"
    phi = FLUID2_VOL / (FLUID1_VOL + FLUID2_VOL)
    assert s["Dispersed fraction phi (initial) [m^3/m^3]"] == pytest.approx(phi)
    assert s["Tip speed (agitator) [m/s]"] == pytest.approx(np.pi * n * d)
    assert s["Reynolds number Re (agitator) [N/N]"] == pytest.approx(n * d ** 2 / 1e-6)
    we = 1000 * n ** 2 * d ** 3 / 0.02
    assert s["Weber number We (agitator) [N/N]"] == pytest.approx(we)
    assert s["d32 correlation (agitator) [m]"] == pytest.approx(
        d * 0.053 * (1 + 3 * phi) * we ** -0.6)


def test_power_metrics_and_tank(stats_case):
    case, stats = stats_case
    (case / "input.xml").write_text(
        "<input><system><dx>0.001</dx><dt>1e-4</dt><domainX>0.2</domainX>"
        "<domainY>0.3</domainY><domainZ>0.2</domainZ></system>"
        "<impeller><name>agitator</name><freq>120</freq><diameter>0.1</diameter>"
        "<rotationAxis>0 1 0</rotationAxis></impeller>"
        "<fluidModel><fluid1Name>Water</fluid1Name><fluid2Name>Oil</fluid2Name>"
        "<rheology><density>1000</density><nu>1e-6</nu></rheology>"
        "<rheology><density>800</density><nu>2e-6</nu></rheology>"
        "<surfaceTension>0.02</surfaceTension></fluidModel></input>")
    t = np.arange(0.0, 4.0001, 0.1)
    omega, torque = 4 * np.pi, 0.5  # 120 rpm, 0.5 N m about y -> P = 2 pi W
    write_table(stats / "MovingBody_agitator.txt",
                ["Time [s]", "Torque X [N-m]", "Torque Y [N-m]", "Torque Z [N-m]",
                 "Angular Velocity [rad/s]"],
                [t, np.full_like(t, 3.0), np.full_like(t, -torque), np.zeros_like(t),
                 np.full_like(t, omega)])
    s = mss.case_settings(case, stats)
    assert s["Tank diameter T (domain width) [m]"] == pytest.approx(0.2)
    v = 1e-3  # Fluid Volume in the fixture
    assert s["Liquid height H = V/(pi T^2/4) [m]"] == pytest.approx(v / (np.pi * 0.01))
    assert s["D/T (agitator) [m/m]"] == pytest.approx(0.5)
    vi = (1000 / 800) ** 0.5 * 800 * 2e-6 * 2.0 * 0.1 / 0.02
    assert s["Viscosity group Vi (agitator) [N/N]"] == pytest.approx(vi)
    rows = by_variable(mss.power_metrics(case, mss.find_stats_files(stats), [], s, 1.0))
    p = omega * torque
    assert rows["Impeller Power P (agitator) [W]"]["mean"] == pytest.approx(p)
    assert rows["Power per Volume P/V (all impellers) [W/m^3]"]["mean"] == pytest.approx(p / v)
    eps_max = p / (1000 * 0.1 ** 3)
    assert rows["Maximum Dissipation Estimate P/(rho_c D^3) (agitator) [W/kg]"]["mean"] \
        == pytest.approx(eps_max)
    po = p / (1000 * 2.0 ** 3 * 0.1 ** 5)
    assert rows["Power Number from Torque P/(rho_c N^3 D^5) (agitator) [W/W]"]["mean"] \
        == pytest.approx(po)
    h = v / (np.pi * 0.01)
    assert rows["Blend Time t95 Correlation HIM eq. 9-1 (agitator) [s]"]["mean"] == pytest.approx(
        5.20 * 0.2 ** 1.5 * h ** 0.5 / (po ** (1 / 3) * 2.0 * 0.1 ** 2))
    assert rows["Kolmogorov Scale at Maximum Dissipation (agitator) [m]"]["mean"] \
        == pytest.approx((1e-18 / eps_max) ** 0.25)


def test_case_settings_simulation_time(stats_case):
    case, stats = stats_case
    (case / "input.xml").write_text("<input><system><dx>0.001</dx><runtime>60</runtime></system></input>")
    write_table(stats / "Timing.txt",
                ["Time [s]", "Iteration [-]", "Cumulative Wall Time [min]"],
                [np.array([1.0, 2.0, 3.0, 4.0]), np.array([10, 20, 30, 40]),
                 np.array([5.0, 10.0, 2.0, 8.0])])  # restart after t = 2 s
    s = mss.case_settings(case, stats)
    assert s["Simulated time (input runtime) [s]"] == 60
    assert s["Simulated time (end) [s]"] == pytest.approx(
        mss.load_stats_table(stats / "Fluid.txt")[1][-1, 0])
    assert s["Iterations [count]"] == 40
    assert s["Wall time (cumulative) [h]"] == pytest.approx(18.0 / 60)


def test_dsd_shape_normal_volume_distribution():
    rng = np.random.default_rng(0)
    d = np.clip(rng.normal(1e-3, 2e-4, 4000), 1e-4, None)
    dsd = [(1.0, di, np.pi / 6 * di ** 3, "Fluid 2") for di in d]
    r = by_variable(mss.dsd_shape(dsd, ["Fluid 2"], 0.5))
    assert 1.0 < r["DSD Volume Mean of d/d32 (Fluid 2 drops) [m/m]"]["mean"] < 1.1
    assert 0.1 < r["DSD Volume Std of d/d32 (Fluid 2 drops) [m/m]"]["mean"] < 0.3
    assert mss.dsd_shape(dsd[:3], ["Fluid 2"], 0.5) == []


def test_ref_case(tmp_path):
    rows = [{"case": c, "file": "Fluid.txt", "variable": "Mean Velocity [m/s]",
             "unit": "m/s", "mean": v} for c, v in (("a", 1.0), ("base", 2.0))]
    mss.write_wide_csv(rows, tmp_path / "m.csv", ["velocity"], ref_case="base")
    assert not (tmp_path / "m.html").exists()  # HTML only with --plot-html
    mss.HTML_PLOTS = True
    try:
        mss.write_wide_csv(rows, tmp_path / "m.csv", ["velocity"], ref_case="base")
    finally:
        mss.HTML_PLOTS = False
    assert "reference case: base" in (tmp_path / "m.html").read_text()


def test_default_run_has_no_dispersion_outputs(stats_case):
    case, _ = stats_case
    mss.main([str(case), "--time", "0.5"])
    with open(case / "stats_summary.csv") as fh:
        methods = {r["steady_method"] for r in csv.DictReader(fh)}
    assert methods == {"user"}
    assert not (case / "vof_structures.csv").exists()


def test_rename_body(tmp_path):
    root = tmp_path / "run"
    (root / "stats_plots").mkdir(parents=True)
    (root / "batch_summary_means.csv").write_text(
        "case,MovingBody_Moving Body: Power Number [W/W],MovingBody_agitator: Power Number [W/W],Impeller Power P (Moving Body) [W]\n"
        "A/agitator_case,3.1,,12\nB/x,,2.9,\n")
    (root / "batch_summary.csv").write_text(
        "case,file,variable,mean\nA/agitator_case,MovingBody_Moving Body.txt,Torque X [N m],1\n")
    (root / "stats_plots" / "MovingBody_Moving_Body.html").write_text("<p>Moving Body: Torque</p>")
    n = mss.migrate_body_names(root)
    rows = list(csv.reader(open(root / "batch_summary_means.csv")))
    assert rows[0] == ["case", "MovingBody_Agitator: Power Number [W/W]", "Impeller Power P (Agitator) [W]"]
    assert rows[1] == ["A/agitator_case", "3.1", "12"] and rows[2] == ["B/x", "2.9", ""]  # merged, case id kept
    assert list(csv.reader(open(root / "batch_summary.csv")))[1][1] == "MovingBody_Agitator.txt"
    assert (root / "stats_plots" / "MovingBody_Agitator.html").read_text() == "<p>Agitator: Torque</p>"
    assert n == 4 and mss.migrate_body_names(root) == 0  # idempotent


def test_addons_only_writes_no_base_outputs(stats_case, vof_case, tmp_path):
    case, _ = stats_case
    for p in (vof_case / "out").rglob("*.vtp"):
        dest = case / p.relative_to(vof_case)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    write_two_fluid_vti(case / "Output" / "Volume" / "block0" / "Volume.1.00000e+00.vti",
                        1.0, func=lambda c: (c[:, 1] < 0.2).astype(float))
    out = tmp_path / "addon"
    mss.main([str(case), "--time", "0.5", "--dispersion", "--dispersed-fraction", "0.2",
              "--voxel-size", str(VOXEL), "--uniformity", "--disp-time", "0.5",
              "--addons-only", "--plot-html", "--output-root", str(out)])
    names = {p.name for p in out.rglob("*") if p.is_file()}
    assert {"stats_summary_dispersion.csv", "stats_summary_dispersion_means.csv",
            "stats_summary_uniformity.csv", "stats_summary_uniformity_means.csv",
            "dispersion_timeseries.csv", "dispersion.html"} <= names
    assert not {"stats_summary.csv", "stats_summary_means.csv", "stats_summary_settings.csv"} & names
    # the only plots are the dispersion figure and the bar plots of the add-on means tables
    assert {n for n in names if n.endswith(".html")} <= {"dispersion.html", "stats_summary_dispersion_means.html",
                                                        "stats_summary_uniformity_means.html"}
    with pytest.raises(SystemExit):
        mss.main([str(case), "--time", "0.5", "--addons-only", "--output-root", str(tmp_path / "x")])


def test_mass_transfer_two_film_global_averages():
    s = {"Continuous phase (largest initial volume)": "Fluid 1 (Water)",
         "Fluid 1 kinematic viscosity [m^2/s]": 1e-6, "Fluid 1 density [kg/m^3]": 1000.0,
         "Fluid 2 density [kg/m^3]": 800.0, "Surface tension [N/m]": 0.03,
         "Tank diameter T (domain width) [m]": 0.3,
         "Impeller speed (Agitator) [rpm]": 300.0, "Impeller diameter (Agitator) [m]": 0.1}
    rows = [{"file": "derived", "variable": "Power-based Mean Dissipation P/M [W/kg]", "mean": 0.5}]
    kin = {"a_ss": 600.0, "A_ss": 6.0}
    phi, dc, dd, m = 0.2, 1e-9, 5e-10, 4.0
    r = by_variable(mss.mass_transfer(rows, s, kin, phi, dc, dd, m, 10.0, 1.0))
    d32 = 6 * phi / 600.0
    kc = 0.4 * (0.5 * 1e-6) ** 0.25 * (1e-6 / dc) ** -0.5
    assert r["Drop Diameter for Mass Transfer d32 = 6 phi/a_ss [m]"]["mean"] == pytest.approx(d32)
    assert r["Continuous-Phase k_c Lamont-Scott [m/s]"]["mean"] == pytest.approx(kc)
    assert r["Continuous-Phase k_c Lamont-Scott [m/s]"]["steady_method"].startswith("two-film, global averages")
    kd = 17.9 * dd / d32
    koc = 1 / (1 / kc + 1 / (m * kd))
    k = koc * 600.0 * (1 / (1 - phi) + 1 / (m * phi))
    assert r["Overall K_oc a, circulating drops [1/s]"]["mean"] == pytest.approx(koc * 600.0)
    assert r["Extraction Rate Constant k, circulating drops [1/s]"]["mean"] == pytest.approx(k)
    assert r["Extraction Time t90 = ln10/k, circulating drops [s]"]["mean"] == pytest.approx(np.log(10) / k)
    assert r["Initial Transfer Rate N0 = K_oc A_ss dC0, circulating drops [mol/s]"]["mean"] == pytest.approx(koc * 6.0 * 10.0)
    assert r["Dispersed-Phase k_d Rigid Drops (Newman) [m/s]"]["mean"] == pytest.approx(2 * np.pi ** 2 / 3 * dd / d32)
    assert "Continuous-Phase k_c Skelland-Moeti HIM eq. 12-66 (Agitator) [m/s]" in r
    # without m: coefficients only; without diffusivities: nothing
    r2 = by_variable(mss.mass_transfer(rows, s, kin, phi, dc, dd, None, None, 1.0))
    assert "Continuous-Phase k_c Lamont-Scott [m/s]" in r2 and not any(v.startswith("Overall") for v in r2)
    assert mss.mass_transfer(rows, s, kin, phi, None, None, m, None, 1.0) == []


# ---------------------------------------------------------------- dispersion assessment

ASSESS_SETTINGS = {
    "Continuous phase (largest initial volume)": "Fluid 2 (MEK)",
    "Dispersed fraction phi (initial) [m^3/m^3]": 0.32,
    "Fluid 1 density [kg/m^3]": 997.0, "Fluid 1 dynamic viscosity [Pa s]": 7e-4,
    "Fluid 2 density [kg/m^3]": 800.0, "Fluid 2 dynamic viscosity [Pa s]": 4.05e-4,
    "Fluid 2 kinematic viscosity [m^2/s]": 5.0625e-7, "Surface tension [N/m]": 1e-3,
    "Impeller speed (Agitator) [rpm]": 139.0, "Impeller diameter (Agitator) [m]": 0.405,
    "Fluid volume (total, initial) [m^3]": 0.24, "Simulated time (end) [s]": 40.0,
}


def test_settling_velocity_stokes_limit():
    d, drho, rho, mu = 20e-6, 200.0, 1000.0, 1e-3
    assert mss.settling_velocity(d, drho, rho, mu) == pytest.approx(drho * 9.81 * d ** 2 / (18 * mu), rel=1e-2)
    # large drops: drag above Stokes, so slower than the Stokes formula
    assert mss.settling_velocity(5e-3, drho, rho, mu) < drho * 9.81 * 5e-3 ** 2 / (18 * mu)


def test_dispersion_assessment(tmp_path):
    s = ASSESS_SETTINGS
    rows = [mss.value_row("derived", "Maximum Dissipation Estimate P/(rho_c D^3) (Agitator) [W/kg]", 1.23, "x"),
            mss.value_row("Fluid.txt", "Mean Velocity [m/s]", 0.9, "x"),
            mss.value_row("x", "Sauter Diameter d32 (VOF, Fluid 1 drops) [m]", 9e-3, "x"),
            mss.value_row("x", "Segregation Index S (Fluid 1, 10x3) [m^3/m^3]", 0.47, "x")]
    out = tmp_path / "dispersion_assessment.csv"
    r = by_variable(mss.dispersion_assessment(rows, s, {"a_ss": 72.4}, 0.32, 1.333e-3, 10.0, 4.0,
                                              out_csv=out))
    n = 139 / 60
    we = 800.0 * n ** 2 * 0.405 ** 3 / 1e-3
    d_inv = 0.405 * 0.053 * (1 + 3 * 0.32) * we ** -0.6
    vi = (800 / 997) ** 0.5 * 7e-4 * n * 0.405 / 1e-3
    d_ref = d_inv * (1 + 0.92 * vi ** 0.84) ** 0.6
    assert r["d32 Correlation HIM eq. 12-30 (Agitator) [m]"]["mean"] == pytest.approx(d_inv)
    assert r["d32 Reference Estimate [m]"]["mean"] == pytest.approx(d_ref)
    assert r["d32 Reference / Voxel Size [m/m]"]["mean"] == pytest.approx(d_ref / 1.333e-3)
    assert r["Drop Resolution"]["mean"] == "sub-grid (grid-limited)"
    assert r["Specific Area at d32 Reference a = 6 phi/d32 [1/m]"]["mean"] == pytest.approx(6 * 0.32 / d_ref)
    assert r["Max Specific Area Representable on Grid [1/m]"]["mean"] == pytest.approx(6 * 0.32 / (4 * 1.333e-3))
    assert r["Grid Refinement for 10 Cells per d32 [m/m]"]["mean"] == pytest.approx(10 * 1.333e-3 / d_ref)
    assert r["Dispersion State"]["mean"] == "dispersed, non-uniform"
    assert r["Best-Estimate Specific Area [1/m]"]["mean"] == pytest.approx(6 * 0.32 / d_ref)
    assert r["Impeller Turns Simulated N t [count]"]["mean"] == pytest.approx(n * 40)
    assert r["Transient d32 Factor (3800/(N t))^(1/6) HIM eq. 12-33 [m/m]"]["mean"] == \
        pytest.approx(38 ** (1 / 6))  # N t = 93 is clipped to the correlation range (>= 100)
    assert r["CFD d32 / d32 Reference [m/m]"]["mean"] == pytest.approx(9e-3 / d_ref)
    assert r["Settling / Mean Velocity, CFD d32 Drops [(m/s)/(m/s)]"]["mean"] > \
        100 * r["Settling / Mean Velocity, d32 Reference Drops [(m/s)/(m/s)]"]["mean"]
    with open(out) as fh:
        checks = list(csv.DictReader(fh))
    assert {"check", "value", "unit", "assessment", "basis"} == set(checks[0])
    # Estimate A (free layer + uniform dispersion from S) and Estimate B (a_ss dx)
    r_s = 0.47 ** 2 * 0.32 / 0.68
    f_d = 1 - r_s / (1 + r_s) / 0.32
    assert r["Dispersed Share f_d = 1 - x/phi, Estimate A [m^3/m^3]"]["mean"] == pytest.approx(f_d)
    assert r["Effective Specific Area a_eff = f_d a_ref, Estimate A [1/m]"]["mean"] == \
        pytest.approx(f_d * 6 * 0.32 / d_ref)
    assert r["Grid-Normalised CFD Area a_ss dx, Estimate B [m/m]"]["mean"] == pytest.approx(72.4 * 1.333e-3)
    assert "Lamont-Scott Velocity Scale (eps nu_c)^(1/4) [m/s]" not in r  # no power-based eps

    # partially dispersed: Estimate A is the best estimate; with diffusivities and m, K_oc at d32_ref
    rows[-1] = mss.value_row("x", "Segregation Index S (Fluid 1, 10x3) [m^3/m^3]", 0.7214, "x")
    rows.insert(0, mss.value_row("derived", "Power-based Mean Dissipation P/M [W/kg]", 0.115, "x"))
    dc, dd, m = 1e-9, 2e-9, 0.5
    r = by_variable(mss.dispersion_assessment(rows, s, {"a_ss": 39.7}, 0.3125, 1.333e-3, 10.0,
                                              diff_c=dc, diff_d=dd, m=m))
    r_s = 0.7214 ** 2 * 0.3125 / 0.6875
    f_d = 1 - r_s / (1 + r_s) / 0.3125
    assert f_d == pytest.approx(0.388, abs=1e-3)  # case 12 of the TAK-360 study
    assert r["Dispersion State"]["mean"] == "partially dispersed"
    a_ref = r["Specific Area at d32 Reference a = 6 phi/d32 [1/m]"]["mean"]
    assert r["Best-Estimate Specific Area [1/m]"]["mean"] == pytest.approx(f_d * a_ref)
    nu = 5.0625e-7
    assert r["Lamont-Scott Velocity Scale (eps nu_c)^(1/4) [m/s]"]["mean"] == pytest.approx((0.115 * nu) ** 0.25)
    d_ref = r["d32 Reference Estimate [m]"]["mean"]
    k_c = 0.4 * (0.115 * nu) ** 0.25 * (nu / dc) ** -0.5
    koc = 1 / (1 / k_c + 1 / (m * mss.SH_NEWMAN * dd / d_ref))
    assert r["Overall K_oc at d32 Reference, rigid drops [m/s]"]["mean"] == pytest.approx(koc)
    assert r["Continuous-Film Resistance Share at d32 Reference, rigid drops [(s/m)/(s/m)]"]["mean"] == \
        pytest.approx(koc / k_c)
    assert r["Overall K_oc a_eff, Estimate A, rigid drops [1/s]"]["mean"] == pytest.approx(koc * f_d * a_ref)

    # a free layer: the CFD area is the estimate
    rows[-1] = mss.value_row("x", "Segregation Index S (Fluid 1, 10x3) [m^3/m^3]", 0.9, "x")
    r = by_variable(mss.dispersion_assessment(rows, s, {"a_ss": 8.2}, 0.32, 1.333e-3, 10.0))
    assert r["Dispersion State"]["mean"] == "stratified (free layer)"
    assert r["Best-Estimate Specific Area [1/m]"]["mean"] == pytest.approx(8.2)


def test_restart_snapshots_and_malformed_rows(stats_case, capsys):
    _, stats = stats_case
    full = (stats / "Fluid.txt").read_text()
    for sub in ("run1", "ControlVolume_Top"):  # M-Star restart snapshot / a real sub-folder
        (stats / sub).mkdir()
        (stats / sub / "Fluid.txt").write_text(full)
    names = [f.relative_to(stats).as_posix() for f in mss.find_stats_files(stats)]
    assert names == ["ControlVolume_Top/Fluid.txt", "Fluid.txt", "Interface.txt"]

    lines = full.splitlines()
    n_rows, n_cols = len(lines) - 1, lines[0].count("\t") + 1
    lines.insert(5, "0.35\t1e-3\t0.1")           # garbled row
    lines.append("4.1\t1e-3\t0.")                # half-written last line of a running solve
    (stats / "Fluid.txt").write_text("\n".join(lines) + "\n")
    header, data = mss.load_stats_table(stats / "Fluid.txt")
    assert data.shape == (n_rows, n_cols) and data[-1, 0] == pytest.approx(4.0)
    assert "Stats/Fluid.txt: 2 of 43 row(s) skipped" in capsys.readouterr().err


def test_steady_start_tolerates_short_excursions():
    t = np.arange(0.0, 60.0001, 0.05)
    rise = 1 - np.exp(-t / 2)
    clean = mss.detect_steady_start(t, rise, 1.0, 0.02)
    assert 5 < clean < 8
    bump = rise + np.where((t > 55) & (t <= 55.5), 0.045, 0)  # window means differ for < 1 window
    assert mss.detect_steady_start(t, bump, 1.0, 0.02) == clean
    late = rise + np.where(t > 59.6, 0.3, 0)                  # disturbance at the very end
    assert mss.detect_steady_start(t, late, 1.0, 0.02) == clean
    shift = rise * np.where(t > 50, 1.10, 1.0)                # a real change of level
    assert mss.detect_steady_start(t, shift, 1.0, 0.02) > 50
    # a plateau shorter than 10 windows is too short to judge excursions: every sample must pass
    short = t <= 14
    assert mss.detect_steady_start(t[short], rise[short], 1.0, 0.02) == clean
    bump = rise[short] + np.where((t[short] > 9) & (t[short] <= 9.5), 0.045, 0)
    assert mss.detect_steady_start(t[short], bump, 1.0, 0.02) > 9
    assert mss.detect_steady_start(np.array([]), np.array([]), 1.0, 0.02) is None


def test_negative_impeller_speed_is_a_direction(stats_case):
    case, stats = stats_case
    xml = ("<input><system><dx>0.001</dx></system><impeller><name>agitator</name><freq>{}</freq>"
           "<diameter>0.1</diameter></impeller><fluidModel><fluid1Name>Water</fluid1Name>"
           "<rheology><density>1000</density><nu>1e-6</nu></rheology></fluidModel></input>")
    (case / "input.xml").write_text(xml.format(120))
    fwd = mss.case_settings(case, stats)
    (case / "input.xml").write_text(xml.format(-120))
    rev = mss.case_settings(case, stats)
    assert rev["Impeller speed (agitator) [rpm]"] == -120
    for key in ("Tip speed (agitator) [m/s]", "Reynolds number Re (agitator) [N/N]",
                "Froude number Fr (agitator) [(m/s^2)/(m/s^2)]"):
        assert rev[key] == pytest.approx(fwd[key]) and rev[key] > 0


def test_incomplete_frames_are_skipped(vof_case, tmp_path, capsys):
    """The output step a running solve is still writing must not enter the statistics."""
    d = vof_case / "out" / "Output" / "VolumeVOFSurface" / "block0"
    whole = (d / "VOFSurface.2.00000e+00.vtp").read_bytes()
    (d / "VOFSurface.3.00000e+00.vtp").write_bytes(whole[:int(0.6 * len(whole))])
    _, series = mss.vof_structures(vof_case, min_voxels=8, voxel_size=VOXEL, t_start=0.5, fluids=fluids())
    np.testing.assert_allclose(series["t"], [1.0, 2.0])
    assert "1 incomplete frame(s) skipped" in capsys.readouterr().err

    # a time step that only one of two blocks has written yet
    vol = tmp_path / "multi" / "Volume"
    for block, times in (("block0", (1, 2)), ("block1", (1,))):
        for t in times:
            (vol / block).mkdir(parents=True, exist_ok=True)
            (vol / block / f"Volume.{t:.5e}.vti").write_bytes(b"<VTKFile>\n</VTKFile>\n")
    frames = mss.complete_frames(mss.find_frames(tmp_path / "multi", ".vti"), "volume output")
    assert [(t, len(paths)) for t, paths in frames] == [(1.0, 2)]


def test_batch_survives_a_failing_case(stats_case, tmp_path, capsys):
    _, stats = stats_case
    root = tmp_path / "batch"
    for name in ("a", "b"):
        (root / name / "Stats").mkdir(parents=True)
        for f in ("Fluid.txt", "Interface.txt"):
            (root / name / "Stats" / f).write_text((stats / f).read_text())
    (root / "a" / "input.xml").write_text("<input><system><dx>0.001</dx></system></input>")
    proj = tmp_path / "proj"   # case b has no <dx>: --dispersion cannot run for it
    mss.main([str(root), "--batch", "--time", "0.5", "--dispersion", "--output-root", str(proj)])
    with open(proj / "batch_summary.csv") as fh:
        assert {r["case"] for r in csv.DictReader(fh)} == {"a"}
    err = capsys.readouterr().err
    assert "b: failed" in err and "1 of 2 case(s) are missing from the tables: failed: b" in err


# ---------------------------------------------------------------- reuse, rename, discovery, timing

@pytest.mark.parametrize("batch", [False, True])
def test_reuse_results_of_earlier_runs(stats_case, vof_case, tmp_path, batch):
    """Once the VTK output is deleted, the drop and uniformity results come from the earlier
    sibling *_dispersion run: also for a single-case run (the app: no --batch), and the
    tag keeps naming the run that computed them."""
    case, _ = stats_case
    write_two_fluid_vti(case / "out" / "Output" / "Volume" / "block0" / "Volume.1.00000e+00.vti",
                        1.0, func=lambda c: (c[:, 1] < 0.2).astype(float))
    runs = tmp_path / "runs"
    args = ([str(tmp_path), "--batch", "--cases", "case"] if batch else [str(case)]) \
        + ["--time", "0.5", "--dispersion", "--voxel-size", str(VOXEL), "--uniformity",
           "--disp-time", "0.5", "--addons-only", "--output-root"]
    mss.main(args + [str(runs / "001_dispersion")])
    shutil.rmtree(case / "out")
    for run in ("002_dispersion", "003_dispersion"):
        mss.main(args + [str(runs / run)])
        per_case = runs / run / "case" if batch else runs / run
        for group, file in (("dispersion", "vof_structures.csv"), ("uniformity", "phase_uniformity.csv")):
            with open(per_case / f"stats_summary_{group}.csv") as fh:
                rows = [r for r in csv.DictReader(fh) if r["file"] == file]
            assert rows and all(r["steady_method"] == "user (reused: 001_dispersion)" for r in rows)
            first = runs / "001_dispersion" / per_case.relative_to(runs / run) / file
            assert (per_case / file).read_bytes() == first.read_bytes()


def test_rename_body_touches_only_names(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "table.csv").write_text("x,value,value,,\n1,2,3,4,5\n")  # equal headers, no body name
    (root / "batch_summary_settings.csv").write_text(
        "case,input_xml,Impeller speed (agitator) [rpm],Impeller speed (Moving Body) [rpm]\n"
        "A/07_agitator_run,/data/x/A/07_agitator_run/input.xml,100,\n")
    (root / "stats_summary.csv").write_text(
        "file,variable,steady_method,mean\nMovingBody_agitator.txt,Torque X [N-m],auto,1\n"
        "derived,Impeller Power P (agitator) [W],of 07_agitator_run,2\n")
    assert mss.migrate_body_names(root) == 2
    assert (root / "table.csv").read_text() == "x,value,value,,\n1,2,3,4,5\n"
    assert list(csv.reader(open(root / "batch_summary_settings.csv"))) == [
        ["case", "input_xml", "Impeller speed (Agitator) [rpm]"],
        ["A/07_agitator_run", "/data/x/A/07_agitator_run/input.xml", "100"]]
    assert list(csv.reader(open(root / "stats_summary.csv")))[1:] == [
        ["MovingBody_Agitator.txt", "Torque X [N-m]", "auto", "1"],
        ["derived", "Impeller Power P (Agitator) [W]", "of 07_agitator_run", "2"]]
    assert mss.migrate_body_names(root) == 0  # idempotent


def test_stats_dir_of_the_case_wins_over_copies(stats_case, tmp_path):
    _, stats = stats_case
    case = tmp_path / "root" / "c1"
    for sub in ("backup/out/Stats", "out/Stats", "a/b/Stats", "z/Stats"):
        (case / sub).mkdir(parents=True)
        (case / sub / "Fluid.txt").write_text((stats / "Fluid.txt").read_text())
    (case / "input.xml").write_text("<input/>")
    assert mss.locate_stats_dir(case) == case / "out" / "Stats"
    assert mss.find_batch_cases(tmp_path / "root") == [(case, case / "out" / "Stats")]
    shutil.rmtree(case / "out")
    assert mss.locate_stats_dir(case) == case / "z" / "Stats"  # the shallowest
    assert mss.find_batch_cases(tmp_path / "root") == [(case, case / "z" / "Stats")]
    (case / "Stats").mkdir()
    assert mss.locate_stats_dir(case) == case / "Stats"


def test_run_timing_glitch_and_mixed_runs(tmp_path, capsys):
    def timing(name, t, it, wall):
        stats = tmp_path / name / "Stats"
        write_table(stats / "Timing.txt", ["Time [s]", "Iteration [-]", "Cumulative Wall Time [min]"],
                    [np.array(t, float), np.array(it, float), np.array(wall, float)])
        return mss.run_timing(stats)

    # the counter runs on over restarts: the last value is the total
    s = timing("plain", [1, 2, 3], [10, 20, 30], [5, 10, 15])
    assert s == {"Iterations [count]": 30, "Wall time (cumulative) [h]": pytest.approx(15 / 60)}
    assert capsys.readouterr().err == ""
    # a backward step that is no restart (the counter does not start again) is not added
    s = timing("glitch", [1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [5, 10, 9.5, 12, 14])
    assert s["Wall time (cumulative) [h]"] == pytest.approx(14 / 60)
    assert "steps back 1 time(s) without starting again" in capsys.readouterr().err
    # rows of two solver runs written into one file: the largest counter values
    s = timing("mixed", [1, 2, 3, 2.1, 3.1, 2.2], [10, 20, 30, 5, 31, 6], [10, 20, 30, 4, 31, 5])
    assert s == {"Iterations [count]": 31, "Wall time (cumulative) [h]": pytest.approx(31 / 60)}
    assert "Stats/Timing.txt: time goes backwards 2 time(s)" in capsys.readouterr().err


# ---------------------------------------------------------------- VOF surface: seams, phase sampler

def test_vof_block_seam_is_welded(tmp_path):
    """Two GPU blocks cut one drop on a seam plane; their rims differ by a fraction of a voxel
    (M-Star: one block interpolates halo values), far more than round-off."""
    out = tmp_path / "case" / "Output"
    sphere = pv.Sphere(radius=0.01, theta_resolution=40, phi_resolution=40)  # outward
    halves = [sphere.clip("z", value=0.0).triangulate(),
              sphere.clip("z", value=0.0, invert=False).triangulate()]
    rim = np.abs(halves[1].points[:, 2]) < 1e-9
    halves[1].points[rim, :2] += np.random.default_rng(1).uniform(-0.1, 0.1, (rim.sum(), 2)) * VOXEL
    for b, half in enumerate(halves):
        d = out / "VolumeVOFSurface" / f"block{b}"
        d.mkdir(parents=True)
        pv.PolyData(half.points, half.faces).save(d / "VOFSurface.1.00000e+00.vtp")
    _, series = mss.vof_structures(tmp_path / "case", 8, VOXEL, 0.0, fluids())
    assert series["n_regions"][0] == 1 and series["n_open"][0] == 0
    assert series["n_drops_fluid1"][0] == 1
    assert series["d32_m_fluid1"][0] == pytest.approx(0.02, rel=0.01)

    # the weld joins rims of different blocks only: one surface keeps its two open halves
    s = mss.analyze_vof_surface(pv, pv.merge(halves, merge_points=False), voxel_size=VOXEL)
    assert len(s["vol"]) == 2 and not s["closed"].any() and s["n_welded"] == 0
    for b, half in enumerate(halves):
        half.cell_data["block"] = np.full(half.n_cells, b)
    surf = pv.merge(halves, merge_points=False)
    s = mss.analyze_vof_surface(pv, surf, voxel_size=VOXEL)
    assert len(s["vol"]) == 1 and s["closed"][0] and 0 < s["n_welded"] <= rim.sum()
    assert s["vol"][0] == pytest.approx(sphere.volume, rel=2e-3)
    # rims further apart than SEAM_TOL voxels are different vertices
    far = [halves[0].copy(), halves[1].copy()]
    far[1].points[:, 2] += (mss.SEAM_TOL + 0.2) * VOXEL
    s = mss.analyze_vof_surface(pv, pv.merge(far, merge_points=False), voxel_size=VOXEL)
    assert len(s["vol"]) == 2 and not s["closed"].any() and s["n_welded"] == 0


def test_phase_sampler_uses_the_volume_output(tmp_path):
    """A BoundaryConditions .vti (or any output without the fraction array) written at the
    time of a VOF frame must not stand in for the Volume output."""
    out = tmp_path / "case" / "Output"
    surf = pv.Sphere(radius=0.01, theta_resolution=60, phi_resolution=60)  # outward: looks like fluid 1
    (out / "VolumeVOFSurface" / "block0").mkdir(parents=True)
    surf.save(out / "VolumeVOFSurface" / "block0" / "VOFSurface.0.00000e+00.vtp")
    for kind in ("BoundaryConditions", "Extra", "Volume"):
        img = pv.ImageData(dimensions=(31, 31, 31), spacing=(VOXEL,) * 3, origin=(-0.015,) * 3)
        if kind == "Volume":
            r = np.linalg.norm(img.cell_centers().points, axis=1)
            img.cell_data["Fluid 1 (Water) Volume Fraction (-)"] = (r > 0.01).astype(float)
            img.cell_data["Fluid 2 (Oil) Volume Fraction (-)"] = (r <= 0.01).astype(float)
        else:
            img.cell_data["Boundary Condition"] = np.zeros(img.n_cells)
        (out / kind / "block0").mkdir(parents=True)
        img.save(out / kind / "block0" / f"{kind}.0.00000e+00.vti")
    _, series = mss.vof_structures(tmp_path / "case", 8, VOXEL, 0.0, fluids())
    assert series["n_drops_fluid1"][0] == 0 and series["n_drops_fluid2"][0] == 1


def test_no_area_plateau_uses_the_fallback_window(stats_case, vof_case):
    """a(t) still rising: the VTK statistics use the window the a(t) kinetics fall back to
    (the last max(window, 25 %) of the run), not everything after the flow steady state."""
    case, stats = stats_case
    t = np.arange(0.0, 4.0001, 0.1)
    write_table(stats / "Interface.txt", ["Time [s]", "Area [m^2]"], [t, 1e-3 * (1 + t)])
    surf = vof_surface()
    surf.field_data["TIME"] = np.array([3.5])
    surf.save(case / "out" / "Output" / "VolumeVOFSurface" / "block0" / "VOFSurface.3.50000e+00.vtp")
    mss.main([str(case), "--time", "0.5", "--dispersion", "--voxel-size", str(VOXEL),
              "--sigma", "0.05", "--rho-continuous", "1000"])
    with open(case / "stats_summary.csv") as fh:
        rows = list(csv.DictReader(fh))
    vof = [r for r in rows if r["file"] == "vof_structures.csv"]
    assert vof and all(r["steady_method"] == "a(t) (no plateau, last window)" and r["n_samples"] == "1"
                       and float(r["steady_start_s"]) == pytest.approx(3.0) for r in vof)
    r = by_variable(rows)
    assert r["Plateau Specific Interfacial Area a_ss [1/m]"]["steady_method"] \
        == "a(t) kinetics (no plateau, last window)"
    assert float(r["Plateau Specific Interfacial Area a_ss [1/m]"]["steady_start_s"]) == pytest.approx(3.0)
    hinze = r["Hinze Max Stable Drop Size [m]"]
    assert hinze["steady_method"] == "a(t) (no plateau, last window)" and hinze["n_samples"] == "11"


# ---------------------------------------------------------------- speed: same numbers

def test_steady_flags_equal_the_loop():
    rng = np.random.default_rng(0)
    for n, dt in ((400, 0.05), (3000, 0.01), (60, 0.4)):
        t = np.cumsum(rng.uniform(0.5, 1.5, n)) * dt
        t[n // 2] = t[n // 2 - 1]  # a repeated time
        for v in (1 - np.exp(-8 * t / t[-1]) + 0.01 * rng.standard_normal(n),
                  rng.standard_normal(n).cumsum(), 1e9 + rng.standard_normal(n), np.rint(3 * np.sin(t))):
            for window, tol in ((1.0, 0.02), (0.3, 0.001), (2.5, 0.2)):
                fast = mss.steady_flags(t, v, window, tol, np.abs(v).max())
                assert np.array_equal(fast, mss.steady_flags_loop(t, v, window, tol, np.abs(v).max()))
    assert fast.any() and not fast.all()
    # unsorted times or gaps in the values: the loop decides
    t, v = np.r_[t[100:], t[:100]], np.r_[v[100:], v[:100]]
    v[5] = np.nan
    assert np.array_equal(mss.steady_flags(t, v, 2.5, 0.2, 3.0), mss.steady_flags_loop(t, v, 2.5, 0.2, 3.0))
    # 100 000 samples in a moment (the loop needs about 14 s); 1-s means change by < 2 % from t = 25 s
    t = np.arange(100_000) * 0.01
    assert mss.detect_steady_start(t, 1 - np.exp(-t / 20), 1.0, 0.02) == pytest.approx(24.5, abs=1.0)


def test_stats_table_is_parsed_once(stats_case):
    _, stats = stats_case
    table = mss.load_stats_table(stats / "Fluid.txt")
    assert mss.load_stats_table(stats / "Fluid.txt") is table and not table[1].flags.writeable
    with open(stats / "Fluid.txt", "a") as fh:  # a running solve appends a row
        fh.write("\t".join(["4.1"] + ["1"] * (len(table[0]) - 1)) + "\n")
    grown = mss.load_stats_table(stats / "Fluid.txt")
    assert grown is not table and len(grown[1]) == len(table[1]) + 1


def test_thread_caps_are_set_before_numpy():
    def threads(code, **env):
        env = {**{k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS")}, **env}
        out = subprocess.run(
            [sys.executable, "-c", code + "; import os; print(*(os.environ.get(k + '_NUM_THREADS', '-') "
                                         "for k in ('OMP', 'OPENBLAS', 'MKL')))"],
            env=env, capture_output=True, text=True, check=True, cwd=Path(mss.__file__).parent)
        return out.stdout.split()

    assert threads("import mstar_stats_summary") == ["4", "4", "4"]
    assert threads("import mstar_stats_summary", OMP_NUM_THREADS="2") == ["2", "4", "4"]
    # the M-Star Post app imports the file after numpy: its environment is left alone
    assert threads("import numpy, mstar_stats_summary") == ["-", "-", "-"]


def uniformity_reference(img, names, ax, nz, nr, center=None, solid=None):
    """Zonal statistics per voxel, the straightforward way: [(phi mean, CoV, S)] per array."""
    n = np.array(img.dimensions) - 1
    axes = [img.origin[a] + (np.arange(n[a]) + 0.5) * img.spacing[a] for a in range(3)]
    cc = np.stack([g.ravel() for g in np.meshgrid(*axes[::-1], indexing="ij")][::-1], axis=1)  # x fastest
    phi = np.array([np.asarray(img.cell_data[n], float) for n in names])
    fluid = ~solid if solid is not None else phi.sum(axis=0) > 0.5 if len(names) > 1 \
        else np.ones(len(cc), bool)
    h = [a for a in range(3) if a != ax]
    c = center or [(img.bounds[2 * a] + img.bounds[2 * a + 1]) / 2 for a in h]
    v, r = cc[fluid, ax], np.sqrt((cc[fluid, h[0]] - c[0]) ** 2 + (cc[fluid, h[1]] - c[1]) ** 2)
    d = img.spacing[ax]
    z0, z1 = v.min() - d / 2, v.max() + d / 2
    zone = np.minimum(((v - z0) / (z1 - z0) * nz).astype(int), nz - 1) * nr \
        + np.minimum((r / r.max() * nr).astype(int), nr - 1)
    w = np.bincount(zone, minlength=nz * nr)
    out = []
    for p in phi[:, fluid]:
        m = np.bincount(zone, p, minlength=nz * nr)[w > 0] / w[w > 0]
        mean = np.average(m, weights=w[w > 0])
        std = np.sqrt(np.average((m - mean) ** 2, weights=w[w > 0]))
        out.append((p.mean(), std / mean, std / np.sqrt(mean * (1 - mean))))
    return out


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_phase_uniformity_axes_center_and_blocks(tmp_path, axis):
    """Zone and layer sums from the layer / column indices equal the per-voxel reference for
    every vertical axis, with --axis-center, a solid flag, and a grid split into two blocks."""
    rng = np.random.default_rng("xyz".index(axis))
    img = pv.ImageData(dimensions=(14, 10, 12), spacing=(0.01, 0.02, 0.015), origin=(-0.07, 0.01, -0.1))
    cc = img.cell_centers().points
    phi1 = np.clip(0.5 + 0.4 * np.sin(20 * cc[:, 0]) * np.cos(15 * cc[:, 1]) + 0.3 * cc[:, 2] / 0.1
                   + 0.05 * rng.standard_normal(len(cc)), 0, 1)
    solid = np.hypot(cc[:, 0], cc[:, 2] + 0.02) > 0.085
    f1, f2 = "Fluid 1 (Water) Volume Fraction (-)", "Fluid 2 (Oil) Volume Fraction (-)"
    img.cell_data[f1], img.cell_data[f2] = np.where(solid, 0.0, phi1), np.where(solid, 0.0, 1 - phi1)
    img.field_data["TIME"] = np.array([1.0])
    ax, center = "xyz".index(axis), [0.01234, -0.00417]  # off the grid: no voxel on a zone boundary

    def run(name, write):
        d = tmp_path / name / "Volume"
        write(d)
        rows, series = mss.phase_uniformity(tmp_path / name, 4, 2, 0.0, vertical_axis=axis,
                                            axis_center=center, zone_scales=[(4, 2), (7, 3)])
        with open(tmp_path / name / "phase_uniformity.csv") as fh:
            return next(csv.DictReader(fh)), series

    def one(d):
        (d / "block0").mkdir(parents=True)
        img.save(d / "block0" / "Volume.1.00000e+00.vti")

    def two(d):  # the same grid as two blocks along x
        for b, (i0, i1) in enumerate(((0, 6), (6, 13))):
            part = pv.ImageData(dimensions=(i1 - i0 + 1, 10, 12), spacing=img.spacing,
                                origin=(img.origin[0] + i0 * img.spacing[0], *img.origin[1:]))
            for n in (f1, f2):
                part.cell_data[n] = np.asarray(img.cell_data[n]).reshape(11, 9, 13)[:, :, i0:i1].ravel()
            part.field_data["TIME"] = np.array([1.0])
            (d / f"block{b}").mkdir(parents=True)
            part.save(d / f"block{b}" / "Volume.1.00000e+00.vti")

    profiles = []
    for name, write in (("whole", one), ("blocks", two)):
        rec, series = run(name, write)
        for nz, nr in ((4, 2), (7, 3)):
            ref = uniformity_reference(img, (f1, f2), ax, nz, nr, center)
            for k, (mean, cov, s) in enumerate(ref, 1):
                assert float(rec[f"phi_mean_fluid{k} [m^3/m^3]"]) == pytest.approx(mean, rel=1e-12)
                assert series[f"cov_fluid{k}_{nz}x{nr}"][0] == pytest.approx(cov, rel=1e-10)
                assert series[f"S_fluid{k}_{nz}x{nr}"][0] == pytest.approx(s, rel=1e-10)
        profiles.append([[r[k] for k in ("z_m", "phi_mean_fluid1", "phi_mean_fluid2")] for r in series["profile"]])
    np.testing.assert_allclose(profiles[1], profiles[0], rtol=1e-12)  # block by block: the same sums

    # one fraction array with a solid flag
    single = pv.ImageData(dimensions=img.dimensions, spacing=img.spacing, origin=img.origin)
    single.cell_data["Volume Fraction"], single.cell_data["Solid Flag"] = phi1, solid.astype(np.uint8)
    (tmp_path / "single").mkdir()
    single.save(tmp_path / "single" / "Volume.1.00000e+00.vti")
    _, series = mss.phase_uniformity(tmp_path / "single", 4, 2, 0.0, vertical_axis=axis, axis_center=center)
    (_, cov, _), = uniformity_reference(single, ("Volume Fraction",), ax, 4, 2, center, solid=solid)
    assert series["cov_4x2"][0] == pytest.approx(cov, rel=1e-10)
    layers = {r["z_m"]: r["phi_mean"] for r in series["profile"]}
    v = cc[:, ax]
    for z in np.unique(v.round(9)):
        sel = np.isclose(v, z) & ~solid
        if sel.any():
            assert layers[min(layers, key=lambda k: abs(k - z))] == pytest.approx(phi1[sel].mean(), rel=1e-12)


def test_vof_surface_without_scipy(monkeypatch):
    """scipy speeds the structure search up and welds block seams; without it the VTK
    connectivity gives the same structures in the same order."""
    surf = vof_surface()
    with_scipy = mss.analyze_vof_surface(pv, surf, voxel_size=VOXEL)
    for name in ("scipy", "scipy.sparse", "scipy.sparse.csgraph", "scipy.spatial"):
        monkeypatch.setitem(sys.modules, name, None)
    without = mss.analyze_vof_surface(pv, surf, voxel_size=VOXEL)
    for key in ("vol", "area", "closed", "signed", "inside"):
        np.testing.assert_array_equal(without[key], with_scipy[key])
    surf.cell_data["block"] = np.arange(surf.n_cells) % 2
    assert mss.analyze_vof_surface(pv, surf, voxel_size=VOXEL)["n_welded"] == -1


def _worker_dies(paths):
    os._exit(3)


def test_workers_same_numbers_and_clear_errors(vof_case, tmp_path):
    for w in (1, 2):
        (tmp_path / f"w{w}").mkdir()
    seq = mss.vof_structures(vof_case, 8, VOXEL, 0.5, fluids(), out_dir=tmp_path / "w1")
    par = mss.vof_structures(vof_case, 8, VOXEL, 0.5, fluids(), out_dir=tmp_path / "w2", workers=2)
    assert seq[0] == par[0] and len(seq[1]["t"]) == 2
    for t in (1.0, 2.0, 3.0):
        write_two_fluid_vti(tmp_path / "uni" / "Volume" / "block0" / f"Volume.{t:.5e}.vti", t,
                            func=lambda c, t=t: 0.5 + 0.4 * np.sin(10 * c[:, 1] + t))
    seq = mss.phase_uniformity(tmp_path / "uni", 10, 3, 0.0, out_dir=tmp_path / "w1")
    par = mss.phase_uniformity(tmp_path / "uni", 10, 3, 0.0, out_dir=tmp_path / "w2", workers=3)
    assert seq[0] == par[0]
    for name in ("vof_structures.csv", "vof_dsd.csv", "phase_uniformity.csv", "holdup_profile.csv",
                 "phase_continuity.csv"):
        assert (tmp_path / "w1" / name).read_bytes() == (tmp_path / "w2" / name).read_bytes()

    # a frame that fails, or a worker that is killed, is an error naming the frame (no hang)
    jobs = [([tmp_path / f"VOFSurface.{t}.00000e+00.vtp"], float(t), [], 1, VOXEL, 0.0, [], None) for t in (1, 2)]
    with pytest.raises(RuntimeError, match=r"VOF structures: frame VOFSurface\.1\.00000e\+00\.vtp "
                                           r"failed in a worker: FileNotFoundError"):
        mss.run_frames(mss.vof_frame, jobs, 2, "VOF structures")
    with pytest.raises(RuntimeError, match=r"a worker process died .* frame a\.vtp"):
        mss.run_frames(_worker_dies, [([Path("a.vtp")],), ([Path("b.vtp")],)], 2, "VOF structures")
