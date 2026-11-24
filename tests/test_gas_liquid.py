"""Tests for the free-surface gas-liquid mass transfer of mstar_stats_summary.py (synthetic data)."""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mstar_stats_summary as mss  # noqa: E402

pv = pytest.importorskip("pyvista")

NU, DIFF, RHO = 1e-6, 1e-9, 1000.0
V_LIQ, AREA0, AREA = 0.5, 1.0, 1.2
TORQUE, OMEGA = 2.0, 4 * np.pi          # P = 8 pi W
EPS_BULK = TORQUE * OMEGA / (RHO * V_LIQ)
H = 0.05                                # voxel size of the synthetic Volume output
ALPHA, EPS_SURF = 2.0, 0.3              # velocity gradient and dissipation of its liquid

INPUT_XML = """<input><system><dx>{dx}</dx><domainX>1.0</domainX><domainY>1.0</domainY>
<domainZ>1.0</domainZ></system>
<fluidModel><fluidModelType>{model}</fluidModelType>
<rheology><density>1000</density><nu>1e-6</nu></rheology>
<surfaceTension>0.07</surfaceTension></fluidModel>
<impeller><name>Agitator</name><rotationAxis>0 1 0</rotationAxis><freq>120</freq>
<diameter>0.4</diameter></impeller></input>"""


def write_table(path: Path, header: list[str], columns: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for row in np.column_stack(columns):
            fh.write("\t".join(f"{v:.9g}" for v in row) + "\n")


def by_variable(rows: list[dict]) -> dict[str, dict]:
    return {r["variable"]: r for r in rows}


def make_case(case: Path, model: int = 2, surface_stats: tuple = ("Velocity Mean [m/s]",),
              torque: bool = True) -> Path:
    """A free-surface case: the surface area steps from AREA0 to AREA at t = 1 s."""
    stats = case / "out" / "Stats"
    t = np.arange(0.0, 4.0001, 0.1)
    one = np.ones_like(t)
    write_table(stats / "Fluid.txt",
                ["Time [s]", "Fluid Volume [m^3]", "Mean Velocity [m/s]", "Mean Energy Dissipation Rate [W/kg]"],
                [t, V_LIQ * one, 0.1 * one, 0.02 * one])
    extra = {"Velocity Mean [m/s]": 0.2 * one, "EDR Mean [W/kg]": 0.01 * one}
    write_table(stats / "Interface.txt", ["Time [s]", "Area [m^2]", *surface_stats],
                [t, np.where(t >= 1, AREA, AREA0), *(extra[k] for k in surface_stats)])
    if torque:
        write_table(stats / "MovingBody_Agitator.txt",
                    ["Time [s]", "Torque X [N-m]", "Torque Y [N-m]", "Torque Z [N-m]", "Angular Velocity [rad/s]"],
                    [t, 0 * one, -TORQUE * one, 0 * one, OMEGA * one])
    (case / "input.xml").write_text(INPUT_XML.format(dx=H, model=model))
    return stats


def stats_rows(case: Path, stats: Path, **kw):
    files = mss.find_stats_files(stats)
    s = mss.case_settings(case, stats)
    return mss.free_surface_stats(case, files, stats, s, 1.0, DIFF, NU, kw.pop("renewal", 0.5),
                                  kw.pop("d_imp", 0.4), **kw)


# ---------------------------------------------------------------- approach 1: Stats files

def test_stats_estimate_area_and_bulk_models(tmp_path):
    case = tmp_path / "case"
    stats = make_case(case)
    rows, series = stats_rows(case, stats, out_path=tmp_path / "ts.csv")
    r = by_variable(rows)
    a = AREA / V_LIQ
    assert r["Specific Free-Surface Area a = A/V_liq [1/m]"]["mean"] == pytest.approx(a)
    assert r["Free-Surface Area Enhancement A/A0 [m^2/m^2]"]["mean"] == pytest.approx(AREA / AREA0)
    # the bulk dissipation is the impeller power over the liquid mass, not the resolved Fluid.txt mean
    eps = r[mss.GL_EPS_BULK]
    assert eps["mean"] == pytest.approx(EPS_BULK) and eps["steady_method"] == "power-based P/M"
    sc = NU / DIFF
    scale = (EPS_BULK * NU) ** 0.25
    for name, c, n in (("Lamont-Scott", 0.4, -0.5), ("Kawase-Moo-Young", 0.301, -0.5),
                       ("Kastanek", 2 / np.sqrt(np.pi), -0.5),
                       ("Calderbank-Moo-Young Rigid Surface", 0.13, -2 / 3)):
        kl = c * scale * sc ** n
        assert r[f"kL {name}, bulk dissipation [m/s]"]["mean"] == pytest.approx(kl)
        assert r[f"kLa {name}, bulk dissipation [1/s]"]["mean"] == pytest.approx(kl * a)
    # surface speed of Interface.txt: Higbie with t_c = L / u_s, Thomas with Re = u_s D_imp / nu
    assert r["kL Higbie, surface mean (Stats) [m/s]"]["mean"] == pytest.approx(
        2 * np.sqrt(DIFF * 0.2 / (np.pi * 0.5)))
    assert r["kLa Thomas, surface mean (Stats) [1/s]"]["mean"] == pytest.approx(
        0.19 * DIFF / 0.4 * (0.2 * 0.4 / NU) ** 0.8 * sc ** (1 / 3) * a)
    assert not any("Lamont-Scott, surface mean" in v for v in r)  # no 'EDR Mean' column in this case
    with open(tmp_path / "ts.csv", newline="") as fh:
        table = list(csv.reader(fh))
    assert table[0][0] == "time [s]" and len(table) == 42
    col = table[0].index("kLa Lamont-Scott, bulk dissipation [1/s]")
    # the area step at t = 1 s is in the series: a kLa at every Stats sample
    assert float(table[-1][col]) / float(table[5][col]) == pytest.approx(AREA / AREA0)


def test_stats_estimate_surface_dissipation_and_no_torque(tmp_path):
    case = tmp_path / "case"
    stats = make_case(case, surface_stats=("EDR Mean [W/kg]",), torque=False)
    r = by_variable(stats_rows(case, stats)[0])
    # without a torque record the bulk dissipation is the resolved mean of Fluid.txt
    assert r[mss.GL_EPS_BULK]["mean"] == pytest.approx(0.02)
    assert "resolved volume mean" in r[mss.GL_EPS_BULK]["steady_method"]
    assert r["kL Lamont-Scott, surface mean (Stats) [m/s]"]["mean"] == pytest.approx(
        0.4 * (0.01 * NU) ** 0.25 * (NU / DIFF) ** -0.5)
    assert not any(v.startswith("kL Higbie") for v in r)  # no surface speed in Interface.txt


# ---------------------------------------------------------------- approach 2: VTK output

def liquid_volume(blocks: int = 1) -> list:
    """20^3 cells of size H; liquid below y = 0.5 with u = ALPHA (x, -2 y, z) and the
    dissipation EPS_SURF, gas above with zeros (as M-Star writes it); split in x for 2 blocks."""
    out = []
    nx = 20 // blocks
    for b in range(blocks):
        img = pv.ImageData(dimensions=(nx + 1, 21, 21), spacing=(H, H, H), origin=(b * nx * H, 0, 0))
        c = img.cell_centers().points
        liq = c[:, 1] < 0.5
        img.cell_data["Velocity Vector (m/s)"] = \
            (ALPHA * c * [1, -2, 1] * liq[:, None]).astype(np.float32)
        img.cell_data["Energy Dissipation Rate (W/kg)"] = (EPS_SURF * liq).astype(np.float32)
        img.cell_data["Time-Avg Energy Dissipation Rate (W/kg)"] = np.full(len(c), 9.0, np.float32)
        out.append(img)
    return out


def save_vti(img, path: Path, raw: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not raw:
        img.save(path)
        return
    import vtk
    w = vtk.vtkXMLImageDataWriter()  # M-Star's layout: raw appended data, no compression
    w.SetFileName(str(path))
    w.SetInputData(img)
    w.SetDataModeToAppended()
    w.EncodeAppendedDataOff()
    w.SetCompressorTypeToNone()
    w.SetHeaderTypeToUInt64()
    w.Write()


def flat_surface(normal: float = 1.0, point_data: bool = False):
    """The free surface y = 0.5 of liquid_volume(), padded with degenerate triangles."""
    s = pv.Plane(center=(0.5, 0.5, 0.5), direction=(0, normal, 0), i_size=1, j_size=1,
                 i_resolution=10, j_resolution=10).triangulate()
    surf = pv.PolyData(s.points, np.r_[s.faces, [3, 0, 0, 0] * 5])
    if point_data:
        surf.point_data["Energy Dissipation Rate (W/kg)"] = np.full(surf.n_points, 0.2, np.float32)
        surf.point_data["Velocity (m/s)"] = np.tile(np.float32([0.3, 0.0, 0.4]), (surf.n_points, 1))
    return surf


def write_frames(case: Path, times, raw: bool = True, blocks: int = 1, volume_times=None, **surface):
    out = case / "out" / "Output"
    (out / "VolumeVOFSurface" / "block0").mkdir(parents=True, exist_ok=True)
    for t in times:
        flat_surface(**surface).save(out / "VolumeVOFSurface" / "block0" / f"VOFSurface.{t:.5e}.vtp")
    for t in times if volume_times is None else volume_times:
        for b, img in enumerate(liquid_volume(blocks)):
            save_vti(img, out / "Volume" / f"block{b}" / f"Volume.{t:.5e}.vti", raw)


def local_rows(case: Path, t_start: float = 0.0, **kw):
    vol = (np.array([0.0, 10.0]), np.array([V_LIQ, V_LIQ]))
    return by_variable(mss.free_surface_local(case, t_start, H, DIFF, NU, vol, 0.5, 0.4, **kw))


@pytest.mark.parametrize("raw, blocks, normal", [(True, 1, 1.0), (False, 1, -1.0), (True, 2, 1.0)])
def test_local_estimate_from_fields_below_the_surface(tmp_path, raw, blocks, normal):
    """Memory-mapped and pyvista-read volumes, one and two GPU blocks, either sense of the
    surface normals: the liquid side is found and its fields are integrated over the surface."""
    case = tmp_path / "case"
    write_frames(case, [1.0, 2.0], raw=raw, blocks=blocks, normal=normal)
    assert mss.VtiBlock(next(case.rglob("*.vti")))._raw is raw
    r = local_rows(case, out_dir=tmp_path)
    assert r["Free-Surface Area (VOF surface) [m^2]"]["mean"] == pytest.approx(1.0)
    assert r["Sampled Share of the Free-Surface Area (VTK) [m^2/m^2]"]["mean"] == pytest.approx(1.0)
    # the instantaneous dissipation of the liquid, not the gas zeros nor the time average (9)
    assert r[mss.GL_EPS_LOCAL]["mean"] == pytest.approx(EPS_SURF, rel=1e-6)
    kl = 0.4 * (EPS_SURF * NU) ** 0.25 * (NU / DIFF) ** -0.5
    assert r["kL Lamont-Scott, local (VTK) [m/s]"]["mean"] == pytest.approx(kl, rel=1e-6)
    assert r["kLa Lamont-Scott, local (VTK) [1/s]"]["mean"] == pytest.approx(kl * 1.0 / V_LIQ, rel=1e-6)
    # u = ALPHA (x, -2 y, z) under a surface normal to y: divergence in the surface = 2 ALPHA
    assert r["Surface Divergence rms (VTK) [1/s]"]["mean"] == pytest.approx(2 * ALPHA, rel=1e-5)
    assert r["kL Surface Divergence, local (VTK) [m/s]"]["mean"] == pytest.approx(
        0.5 * np.sqrt(DIFF * 2 * ALPHA), rel=1e-5)
    # speed of the cells 1.5 voxels below the surface (layer y = 0.425), at the triangle centroids
    cen = flat_surface(normal).cell_centers().points[:200]  # without the degenerate padding
    cell = (np.floor(cen / H) + 0.5) * H
    speed = ALPHA * np.sqrt(cell[:, 0] ** 2 + (2 * 0.425) ** 2 + cell[:, 2] ** 2)
    assert r["Free-Surface Speed (VTK) [m/s]"]["mean"] == pytest.approx(speed.mean(), rel=1e-5)
    assert r["kL Higbie, local (VTK) [m/s]"]["mean"] == pytest.approx(
        2 * np.sqrt(DIFF / (np.pi * 0.5)) * np.sqrt(speed).mean(), rel=1e-5)
    with open(tmp_path / mss.GL_SURFACE_CSV, newline="") as fh:
        table = list(csv.reader(fh))
    assert table[0][0] == "time [s]" and [row[0] for row in table[1:]] == ["1.0", "2.0"]


def test_local_estimate_uses_one_field_source(tmp_path):
    """Volume output for one of three surface frames: the point data on the surface is used
    for all of them; with a Volume file per frame the cells below the surface are."""
    case = tmp_path / "case"
    write_frames(case, [1.0, 2.0, 3.0], volume_times=[2.0], point_data=True)
    r = local_rows(case)
    assert r[mss.GL_EPS_LOCAL]["mean"] == pytest.approx(0.2, rel=1e-6)
    assert r[mss.GL_EPS_LOCAL]["n_samples"] == 3
    assert r["Free-Surface Speed (VTK) [m/s]"]["mean"] == pytest.approx(0.5, rel=1e-6)
    assert r["Surface Divergence rms (VTK) [1/s]"]["n_samples"] == 1  # needs the Volume output
    write_frames(case, [1.0, 2.0, 3.0], point_data=True)
    assert local_rows(case)[mss.GL_EPS_LOCAL]["mean"] == pytest.approx(EPS_SURF, rel=1e-6)


def test_local_estimate_counts_closed_structures(tmp_path):
    d = tmp_path / "case" / "out" / "Output" / "VolumeVOFSurface" / "block0"
    d.mkdir(parents=True)
    bubble = pv.Sphere(radius=0.1, center=(0.5, 0.2, 0.5), theta_resolution=40, phi_resolution=40).triangulate()
    (flat_surface() + bubble).save(d / "VOFSurface.1.00000e+00.vtp")
    r = local_rows(tmp_path / "case")
    assert r["Closed Structures, Bubbles and Droplets (VOF surface) [count]"]["mean"] == 1
    share = r["Closed-Structure Share of the Free-Surface Area (VOF surface) [m^2/m^2]"]["mean"]
    assert share == pytest.approx(bubble.area / (1.0 + bubble.area), rel=1e-6)
    assert not any(v.startswith("kL") for v in r)  # no fields without Volume output or point data


# ---------------------------------------------------------------- grid needed

def test_assessment_grid_for_a_resolved_calculation():
    eps_s, dx = 0.04, 1e-3
    s = {"LX [cells]": 100, "LY [cells]": 200, "LZ [cells]": 100,
         "Surface tension [N/m]": 0.07, "Fluid 1 density [kg/m^3]": RHO}
    gl = [{"variable": mss.GL_EPS_BULK, "mean": 1.0}, {"variable": mss.GL_EPS_LOCAL, "mean": eps_s},
          {"variable": "kLa Lamont-Scott, local (VTK) [1/s]", "mean": 0.002},
          {"variable": "kLa Kastanek, bulk dissipation [1/s]", "mean": 0.02}]
    base = [{"variable": "Blend Time t95 Correlation HIM eq. 9-1 (Agitator) [s]", "mean": 60.0}]
    r = by_variable(mss.gas_liquid_assessment(base, gl, s, dx, DIFF, NU, 1.0))
    eta = (NU ** 3 / eps_s) ** 0.25  # at the surface dissipation, not the bulk value
    eta_b = eta * (NU / DIFF) ** -0.5
    assert r["Kolmogorov Scale at the Free Surface eta_s [m]"]["mean"] == pytest.approx(eta)
    assert r["Concentration Film Thickness delta_c = D/kL [m]"]["mean"] == pytest.approx(2.5 * eta_b)
    assert r["Voxel Size for Resolved Free-Surface Turbulence [m]"]["mean"] == pytest.approx(2.1 * eta)
    need = 2.5 * eta_b / 10
    refine = dx / need
    assert r["Voxel Size for a Resolved Concentration Film [m]"]["mean"] == pytest.approx(need)
    assert r["Grid Refinement for a Resolved Concentration Film [m/m]"]["mean"] == pytest.approx(refine)
    assert r["Lattice Cost Factor for a Resolved Concentration Film [count]"]["mean"] == pytest.approx(refine ** 4)
    assert r["Lattice Cells for a Resolved Concentration Film [count]"]["mean"] == pytest.approx(2e6 * refine ** 3)
    assert r["Concentration Film Resolution"]["mean"].startswith("subgrid")
    assert r["Free-Surface Turbulence Resolution"]["mean"].startswith("modelled")
    # mixing against the fastest transfer estimate: 60 s blend vs t90 = ln 10 / 0.02 = 115 s
    assert r["Blend Time / Mass Transfer Time t95/t90 [s/s]"]["mean"] == pytest.approx(60 * 0.02 / np.log(10))
    assert r["Liquid Mixing vs Mass Transfer"]["mean"].startswith("mixing and transfer on similar")
    # a grid of 2 eta_s resolves the turbulence, and needs no further refinement for it
    fine = by_variable(mss.gas_liquid_assessment(base, gl, s, 2 * eta, DIFF, NU, 1.0))
    assert fine["Free-Surface Turbulence Resolution"]["mean"] == "resolved (DNS)"
    assert "Lattice Cost Factor for Resolved Free-Surface Turbulence [count]" not in fine


# ---------------------------------------------------------------- command line

def gl_args(*extra):
    return ["--gas-liquid", "free-surface", "--gas-diffusivity", str(DIFF), "--time", "1", *extra]


def read(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_main_gas_liquid_end_to_end(tmp_path):
    case = tmp_path / "case"
    make_case(case)
    write_frames(case, [1.0, 2.0, 3.0])
    out = tmp_path / "out"
    mss.main([str(case), *gl_args("--addons-only", "--output-root", str(out))])
    assert sorted(p.name for p in out.iterdir()) == [
        "gas_liquid_assessment.csv", "gas_liquid_surface.csv", "gas_liquid_timeseries.csv",
        "stats_summary_gas_liquid.csv", "stats_summary_gas_liquid_means.csv"]
    r = {row["variable"]: row for row in read(out / "stats_summary_gas_liquid.csv")}
    assert float(r["Gas Diffusivity D [m^2/s]"]["mean"]) == DIFF
    assert float(r["kLa Lamont-Scott, bulk dissipation [1/s]"]["mean"]) == pytest.approx(
        0.4 * (EPS_BULK * NU) ** 0.25 * (NU / DIFF) ** -0.5 * AREA / V_LIQ)
    assert float(r["kLa Lamont-Scott, local (VTK) [1/s]"]["mean"]) == pytest.approx(
        0.4 * (EPS_SURF * NU) ** 0.25 * (NU / DIFF) ** -0.5 * 1.0 / V_LIQ, rel=1e-6)
    # the grid assessment uses the dissipation found at the surface
    assert float(r["Kolmogorov Scale at the Free Surface eta_s [m]"]["mean"]) == pytest.approx(
        (NU ** 3 / EPS_SURF) ** 0.25, rel=1e-6)
    assert [c["check"] for c in read(out / "gas_liquid_assessment.csv")][:2] == [
        "free-surface turbulence, dx / eta_s", "voxel size for resolved free-surface turbulence"]
    wide = read(out / "stats_summary_gas_liquid_means.csv")
    assert len(wide) == 1 and "kLa Lamont-Scott, local (VTK) [1/s]" in wide[0]


def test_free_surface_model_and_diffusivity_are_required(tmp_path):
    case = tmp_path / "case"
    make_case(case, model=4)
    with pytest.raises(SystemExit, match="free-surface model.*immiscible two-fluid"):
        mss.main([str(case), *gl_args()])
    make_case(case)
    with pytest.raises(SystemExit, match="--gas-diffusivity"):
        mss.main([str(case), "--gas-liquid"])
    assert mss.case_settings(case, case / "out" / "Stats")["Fluid model"] == "free surface"


def test_batch_keeps_the_free_surface_cases(tmp_path, capsys):
    root = tmp_path / "runs"
    make_case(root / "fs")
    make_case(root / "single", model=1)
    out = tmp_path / "out"
    mss.main([str(root), "--batch", *gl_args("--addons-only", "--gas-no-local", "--output-root", str(out))])
    assert "single: failed" in capsys.readouterr().err
    rows = read(out / "batch_summary_gas_liquid.csv")
    assert {r["case"] for r in rows} == {"fs"}
    assert not (out / "fs" / "gas_liquid_surface.csv").exists()  # --gas-no-local
    assert (out / "fs" / "gas_liquid_timeseries.csv").is_file()


def test_local_rows_are_reused_when_the_vtk_output_is_gone(tmp_path):
    root = tmp_path / "runs"
    case = root / "a"
    make_case(case)
    write_frames(case, [1.0, 2.0])
    first, second = tmp_path / "20260101-000000_gas_liquid", tmp_path / "20260102-000000_gas_liquid"
    args = [str(root), "--batch", *gl_args("--addons-only")]
    mss.main([*args, "--output-root", str(first)])
    shutil.rmtree(case / "out" / "Output")
    mss.main([*args, "--output-root", str(second)])
    r = {row["variable"]: row for row in read(second / "batch_summary_gas_liquid.csv")}
    local = r["kLa Lamont-Scott, local (VTK) [1/s]"]
    assert "(reused: 20260101-000000_gas_liquid)" in local["steady_method"]
    assert float(local["mean"]) == pytest.approx(
        0.4 * (EPS_SURF * NU) ** 0.25 * (NU / DIFF) ** -0.5 / V_LIQ, rel=1e-6)
    assert (second / "a" / "gas_liquid_surface.csv").is_file()
    # the grid assessment of the new run still sees the surface dissipation
    assert float(r["Kolmogorov Scale at the Free Surface eta_s [m]"]["mean"]) == pytest.approx(
        (NU ** 3 / EPS_SURF) ** 0.25, rel=1e-6)
