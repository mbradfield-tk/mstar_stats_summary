#!/usr/bin/env python3
"""Steady-state summary of all variables tracked in an M-Star Stats folder.

Scans every tab-delimited stats file (recursively, e.g. Fluid.txt,
MovingBody_*.txt, ControlVolume_*/FieldData.txt), detects a single steady-state
start time from the plateau of the mean fluid velocity in Fluid.txt
(windowed-mean drift criterion), or uses a user-defined start time, and writes
a CSV with the steady-state average and standard deviation of every variable.
For scalar tracers (Scalar_*.txt), also reports the 95% mixed time, i.e. when
the concentration RSD last drops below 5%, measured from tracer injection
(detected as the end of the initial zero-concentration period).

Optional phase-dispersion metrics for Immiscible Two-Fluid simulations
(require pyvista for the VTK-based parts; skipped with a warning otherwise):
  --dispersion  specific interfacial area a = A_int/V_liq and area-based
                d32 = 6*phi/a from the interface stats file; drop size
                distribution, d32/d10/d43, drop count and dispersed fraction
                F_disp = 1 - V_largest/V_total from the Volume VOF Surface
                .vtp output (vof_structures.csv, vof_dsd.csv); with --sigma,
                --rho-continuous and --voxel-size also a Hinze drop-size
                resolution check.
  --uniformity  zonal coefficient of variation of the phase fraction and the
                vertical holdup profile from the 3D Volume .vti output
                (phase_uniformity.csv, holdup_profile.csv).
With --plot, the dispersion results are also plotted to stats_plots/dispersion.html.

Example:
    python mstar_stats_summary.py test              # auto-detect steady state
    python mstar_stats_summary.py test --time 5.0   # steady state = t >= 5 s
    python mstar_stats_summary.py test --plot       # plot all variables
    python mstar_stats_summary.py test --plot "mean velocity" "power number"
    python mstar_stats_summary.py runs --batch      # process every case in runs/
    python mstar_stats_summary.py case --dispersion --dispersed-fraction 0.2 \
        --voxel-size 5e-4 --sigma 0.05 --rho-continuous 998 --plot
    python mstar_stats_summary.py case --uniformity --zones-z 10 --zones-r 3
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[stats_summary] {msg}")


def warn(msg: str) -> None:
    print(f"[stats_summary] WARNING: {msg}", file=sys.stderr)


def fail(msg: str) -> None:
    sys.exit(f"[stats_summary] ERROR: {msg}")


# ---------------------------------------------------------------- discovery

def locate_stats_dir(root: Path) -> Path | None:
    """Accept a case dir, a dir containing Stats/, or the Stats dir itself."""
    if root.is_dir():
        if root.name.lower() == "stats":
            return root
        hits = sorted(p for p in root.rglob("Stats") if p.is_dir())
        if hits:
            return hits[0]
        if any(root.glob("*.txt")):
            return root
    return None


def find_stats_dir(root: Path) -> Path:
    if not root.exists():
        fail(f"path not found: {root}")
    stats_dir = locate_stats_dir(root)
    if stats_dir is None:
        fail(f"no Stats folder (or .txt stats files) found under {root}")
    return stats_dir


def find_stats_files(stats_dir: Path) -> list[Path]:
    return sorted(f for f in stats_dir.rglob("*.txt") if f.is_file())


# ---------------------------------------------------------------- parsing

def load_stats_table(path: Path) -> tuple[list[str], np.ndarray] | None:
    """Return (column headers, data[rows, cols]) or None if not a stats table."""
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
    if len(header) < 2 or not header[0].lower().startswith("time"):
        return None
    try:
        data = np.genfromtxt(path, skip_header=1, delimiter="\t")
    except Exception as e:
        warn(f"{path.name}: unreadable ({e})")
        return None
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] == 0 or data.shape[1] != len(header):
        warn(f"{path.name}: data shape does not match header; skipped")
        return None
    return header, data


def var_unit(name: str) -> str:
    m = re.search(r"\[([^\]]*)\]", name)
    return m.group(1) if m else ""


def norm_name(name: str) -> str:
    """Lowercase and strip the unit suffix, e.g. 'Mean Velocity [m/s]' -> 'mean velocity'."""
    return re.sub(r"\[[^\]]*\]", "", name).strip().lower()


# ---------------------------------------------------------------- steady state

def detect_steady_start(times: np.ndarray, values: np.ndarray,
                        window: float, tol: float) -> float | None:
    """First time from which the windowed mean stops drifting (change < tol) for good.

    Compares the mean of the trailing window against the window before it, which
    tolerates turbulent fluctuations that never settle below tol themselves.
    Returns None if no steady plateau is found.
    """
    scale = np.max(np.abs(values)) if values.size else 0.0
    if scale == 0.0:  # identically zero trace is steady from the start
        return float(times[0])

    steady = np.zeros(len(times), bool)
    for i, t in enumerate(times):
        cur = (times > t - window) & (times <= t)
        prev = (times > t - 2 * window) & (times <= t - window)
        if t < times[0] + 2 * window or cur.sum() < 3 or prev.sum() < 3:
            continue
        m_cur, m_prev = values[cur].mean(), values[prev].mean()
        denom = max(abs(m_prev), tol * scale)
        if abs(m_cur - m_prev) / denom < tol:
            steady[i] = True

    # require steadiness to persist to the end of the trace
    for i in range(len(times)):
        if steady[i] and steady[i:].all():
            # steady window [t-window, t] means the plateau began a window earlier
            return float(max(times[i] - window, times[0]))
    return None


def steady_from_fluid_velocity(files: list[Path], window: float, tol: float) -> float | None:
    """Detect the global steady-state start from the mean velocity in Fluid.txt."""
    fluid = [f for f in files if f.name.lower() == "fluid.txt"]
    if not fluid:
        warn("no Fluid.txt found for velocity-based steady-state detection")
        return None
    table = load_stats_table(fluid[0])
    if table is None:
        return None
    header, data = table
    normed = [norm_name(h) for h in header]
    if "mean velocity" not in normed:
        warn(f"no 'Mean Velocity' column in {fluid[0].name}")
        return None
    col = normed.index("mean velocity")
    times, vals = data[:, 0], data[:, col]
    ok = np.isfinite(times) & np.isfinite(vals)
    log(f"steady-state trace: '{header[col]}' from {fluid[0].name}")
    return detect_steady_start(times[ok], vals[ok], window, tol)


# ---------------------------------------------------------------- summary

def summarize_file(path: Path, rel_name: str, t_start: float, method: str) -> list[dict]:
    table = load_stats_table(path)
    if table is None:
        return []
    header, data = table
    times = data[:, 0]
    rows = []
    for j in range(1, len(header)):
        name = header[j].strip()
        vals = data[:, j]
        ok = np.isfinite(times) & np.isfinite(vals) & (times >= t_start)
        t, vs = times[ok], vals[ok]
        if t.size == 0:
            continue
        rows.append({
            "file": rel_name,
            "variable": name,
            "unit": var_unit(name),
            "steady_method": method,
            "steady_start_s": t_start,
            "end_time_s": float(t[-1]),
            "n_samples": int(vs.size),
            "mean": float(vs.mean()),
            "std": float(vs.std(ddof=1)) if vs.size > 1 else 0.0,
            "min": float(vs.min()),
            "max": float(vs.max()),
        })
    return rows


COLUMNS = ["file", "variable", "unit", "steady_method", "steady_start_s",
           "end_time_s", "n_samples", "mean", "std", "min", "max"]


# ---------------------------------------------------------------- mixing time

def mixing_time_from_rsd(times: np.ndarray, rsd: np.ndarray,
                         threshold: float) -> float | None:
    """Time when RSD last crosses below threshold [%] (linear interpolation).

    Returns None if RSD never exceeds the threshold (no injection resolved)
    or is still above it at the end of the trace (not yet mixed).
    """
    above = rsd >= threshold
    if not above.any() or above[-1]:
        return None
    i = int(np.max(np.nonzero(above)))
    t0, t1, r0, r1 = times[i], times[i + 1], rsd[i], rsd[i + 1]
    return float(t0 + (r0 - threshold) / (r0 - r1) * (t1 - t0))


def injection_start(times: np.ndarray, conc: np.ndarray) -> float | None:
    """Time the tracer is added: last sample at which mean conc is still zero."""
    nz = np.nonzero(conc > 0)[0]
    if nz.size == 0:
        return None
    return float(times[nz[0] - 1]) if nz[0] > 0 else float(times[0])


def scalar_mixing(files: list[Path], stats_dir: Path,
                  threshold: float) -> tuple[list[dict], dict[Path, tuple[float, str]]]:
    """Mixing-time CSV rows for Scalar_*.txt files, plus per-file plot markers.

    Reported mixed times are measured from tracer injection (detected as the
    period while the mean concentration is zero). Returns (rows,
    {file: (absolute mixed time, label)}) for plotting.
    """
    rows, vlines = [], {}
    for f in files:
        if not f.name.lower().startswith("scalar_"):
            continue
        table = load_stats_table(f)
        if table is None:
            continue
        header, data = table
        rel = str(f.relative_to(stats_dir))
        times = data[:, 0]
        normed = [norm_name(h) for h in header]

        t_inj = None
        if "conc mean" in normed:
            conc = data[:, normed.index("conc mean")]
            ok = np.isfinite(times) & np.isfinite(conc)
            t_inj = injection_start(times[ok], conc[ok])
        if t_inj is None:
            t_inj = float(times[0])
            warn(f"  {rel}: could not detect tracer injection "
                 f"(no nonzero 'Conc Mean'); assuming injection at t = {t_inj:g} s")
        else:
            log(f"  {rel}: tracer injection detected at t = {t_inj:.3f} s")
        common = {"file": rel, "unit": "s", "steady_start_s": "",
                  "end_time_s": float(times[-1]), "n_samples": len(times),
                  "std": "", "min": "", "max": ""}
        rows.append({**common, "variable": "tracer injection time",
                     "steady_method": "conc>0", "mean": t_inj})

        for j in range(1, len(header)):
            if "rsd" not in normed[j]:
                continue
            rsd = data[:, j]
            ok = np.isfinite(times) & np.isfinite(rsd)
            t_mix_abs = mixing_time_from_rsd(times[ok], rsd[ok], threshold)
            label = (f"{100 - threshold:g}% mixed time from injection "
                     f"('{header[j].strip()}' < {threshold:g}%)")
            if t_mix_abs is None:
                warn(f"  {rel}: {label}: RSD never crosses below "
                     f"{threshold:g}% (or never exceeds it); skipped")
                continue
            t_mix = t_mix_abs - t_inj
            log(f"  {rel}: {label} = {t_mix:.3f} s (at t = {t_mix_abs:.3f} s)")
            rows.append({**common, "variable": label,
                         "steady_method": "rsd", "mean": t_mix})
            vlines.setdefault(f, (t_mix_abs,
                                  f"{100 - threshold:g}% mixed "
                                  f"({t_mix:.3g} s after injection)"))
    return rows, vlines


def write_csv(rows: list[dict], out_path: Path, columns: list[str] = COLUMNS,
              what: str = "variables") -> None:
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {out_path} ({len(rows)} {what})")


DEFAULT_MEANS_VARS = [
    "energy dissipation", "velocity", "vorticity", "mixed time", "shear",
    "power number", "fluid volume", "viscosity", "density",
    "rotation speed", "angular velocity",
    "interfacial area", "d32", "dispersed fraction", "n_drops", "cov",
]
# with the default list, keep only mean values: drop LB and min/max variants
DEFAULT_MEANS_EXCLUDE = r"\b(lb|min|max)\b"


def write_wide_csv(batch_rows: list[dict], out_path: Path, queries: list[str],
                   warn_unmatched: bool = True, exclude: str | None = None) -> None:
    """One row per case, one 'variable [units]' column per variable, values = means.

    Only variables fuzzy-matching one of the queries (and not the exclude
    regex) are included. Variables tracked in more than one stats file are
    prefixed with the file name to keep columns unique.
    """
    qn = [q.strip().lower() for q in queries]
    rows = [r for r in batch_rows
            if any(q in norm_name(r["variable"]) for q in qn)
            and not (exclude and re.search(exclude, norm_name(r["variable"])))]
    if warn_unmatched:
        for q in qn:
            if not any(q in norm_name(r["variable"]) for r in batch_rows):
                warn(f"means CSV: '{q}' matched no variables")
    if not rows:
        warn(f"means CSV: no variables matched; skipping {out_path}")
        return

    def base_name(r: dict) -> str:
        name = r["variable"]
        if r["unit"] and f"[{r['unit']}]" not in name:
            name = f"{name} [{r['unit']}]"
        return name

    files_per_var: dict[str, set] = {}
    for r in rows:
        files_per_var.setdefault(base_name(r), set()).add(r["file"])

    def col_name(r: dict) -> str:
        name = base_name(r)
        if len(files_per_var[name]) > 1:
            stem = str(Path(r["file"]).with_suffix(""))
            name = f"{stem}: {name}"
        return name

    cols: list[str] = []
    data: dict[str, dict[str, object]] = {}
    for r in rows:
        c = col_name(r)
        if c not in cols:
            cols.append(c)
        data.setdefault(r["case"], {})[c] = r["mean"]

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case"] + cols, restval="")
        w.writeheader()
        for case, vals in data.items():
            w.writerow({"case": case, **vals})
    log(f"wrote {out_path} ({len(data)} cases x {len(cols)} variables)")

    plot_means_bars(cols, data, out_path.with_suffix(".html"))


def plot_means_bars(cols: list[str], data: dict[str, dict[str, object]],
                    out_html: Path) -> None:
    """Bar plots of the means table: absolute values and % difference vs the
    first case (reference), one row of subplots per variable."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    cases = list(data)
    ref = cases[0]
    titles = []
    for c in cols:
        titles += [c, "% diff vs reference"]
    n = len(cols)
    fig = make_subplots(rows=n, cols=2, subplot_titles=titles,
                        vertical_spacing=min(0.6 / n, 0.3),
                        horizontal_spacing=0.08)

    for i, col in enumerate(cols, 1):
        ys = [v if isinstance(v, (int, float)) else np.nan
              for v in (data[case].get(col) for case in cases)]
        fig.add_trace(go.Bar(x=cases, y=ys, marker_color="steelblue",
                             showlegend=False,
                             hovertemplate="%{x}: %{y:.5g}<extra></extra>"),
                      row=i, col=1)
        ref_v = ys[0]
        if np.isfinite(ref_v) and ref_v != 0:
            rel = [(y - ref_v) / abs(ref_v) * 100 for y in ys]
            fig.add_trace(go.Bar(x=cases, y=rel, marker_color="indianred",
                                 showlegend=False,
                                 hovertemplate="%{x}: %{y:.3g}%<extra></extra>"),
                          row=i, col=2)
            fig.update_yaxes(title_text="%", title_font_size=10, row=i, col=2)

    fig.update_xaxes(tickfont_size=9, automargin=True)
    fig.update_annotations(font_size=10)
    fig.update_layout(
        title=f"Steady-state means by case (reference case: {ref})",
        height=max(400, 300 * n), template="plotly_white", margin=dict(t=90))
    fig.write_html(str(out_html))
    log(f"wrote {out_html} ({n} variables, {len(cases)} cases)")


# ---------------------------------------------------------------- dispersion

def import_pyvista(what: str):
    """Return the pyvista module, or None (with a warning) if not installed."""
    try:
        import pyvista as pv
    except ImportError:
        warn(f"pyvista is not installed (pip install pyvista); skipping {what}")
        return None
    return pv


def series_row(file: str, variable: str, t, v, t_start: float,
               method: str) -> dict | None:
    """Steady-state (t >= t_start) summary row of a derived time series."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    ok = np.isfinite(t) & np.isfinite(v) & (t >= t_start)
    if not ok.any():
        warn(f"  {file}: no finite '{variable}' data at t >= {t_start:g} s")
        return None
    ts, vs = t[ok], v[ok]
    return {
        "file": file, "variable": variable, "unit": var_unit(variable),
        "steady_method": method, "steady_start_s": t_start,
        "end_time_s": float(ts[-1]), "n_samples": int(vs.size),
        "mean": float(vs.mean()),
        "std": float(vs.std(ddof=1)) if vs.size > 1 else 0.0,
        "min": float(vs.min()), "max": float(vs.max()),
    }


def fluid_volume_series(files: list[Path], times: np.ndarray) -> np.ndarray | None:
    """'Fluid Volume' from Fluid.txt, interpolated onto times."""
    fluid = [f for f in files if f.name.lower() == "fluid.txt"]
    table = load_stats_table(fluid[0]) if fluid else None
    if table is None:
        return None
    header, data = table
    normed = [norm_name(h) for h in header]
    if "fluid volume" not in normed:
        return None
    j = normed.index("fluid volume")
    ok = np.isfinite(data[:, 0]) & np.isfinite(data[:, j])
    log(f"liquid volume: '{header[j].strip()}' from {fluid[0].name}")
    return np.interp(times, data[ok, 0], data[ok, j])


def interface_metrics(files: list[Path], stats_dir: Path, t_start: float,
                      phi: float | None, v_liq: float | None) -> tuple[list[dict], dict]:
    """Specific interfacial area a = A_int/V_liq and area-based d32 = 6*phi/a.

    Returns (summary rows, {"t", "a"[, "d32"]} time series).
    """
    hits = [f for f in files if "interface" in f.name.lower()]
    if not hits:
        warn("--dispersion: no interface stats file (name containing "
             "'interface') found; skipping interfacial-area metrics")
        return [], {}
    f = hits[0]
    rel = str(f.relative_to(stats_dir))
    table = load_stats_table(f)
    if table is None:
        warn(f"--dispersion: {rel} is not a stats table; skipping interfacial-area metrics")
        return [], {}
    header, data = table
    normed = [norm_name(h) for h in header]
    cols = [j for j in range(1, len(header)) if "area" in normed[j]]
    if not cols:
        warn(f"--dispersion: no interfacial area column in {rel}; available: "
             + ", ".join(h.strip() for h in header[1:]))
        return [], {}
    j = cols[0]
    log(f"interfacial area: '{header[j].strip()}' from {rel}")
    t, area = data[:, 0], data[:, j]

    if v_liq is not None:
        vol = np.full_like(t, v_liq)
        log(f"liquid volume: {v_liq:g} m^3 (--liquid-volume)")
    else:
        vol = fluid_volume_series(files, t)
        if vol is None:
            warn("--dispersion: no 'Fluid Volume' in Fluid.txt; give "
                 "--liquid-volume; skipping interfacial-area metrics")
            return [], {}
    a = np.full_like(t, np.nan)
    np.divide(area, vol, out=a, where=vol > 0)
    series = {"t": t, "a": a}
    rows = [series_row(rel, "Specific Interfacial Area [1/m]", t, a, t_start, "interface")]

    if phi is None:
        warn("--dispersion: --dispersed-fraction not given; skipping area-based d32")
    else:
        d32 = np.full_like(a, np.nan)
        np.divide(6.0 * phi, a, out=d32, where=a > 0)
        series["d32"] = d32
        rows.append(series_row(rel, "Sauter Diameter d32 (interface) [m]",
                               t, d32, t_start, "interface"))
    rows = [r for r in rows if r]
    for r in rows:
        log(f"  {r['variable']}: {r['mean']:.5g} \u00b1 {r['std']:.3g}")
    return rows, series


TIME_IN_NAME = re.compile(r"\.([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\.vt[ip]$")


def find_frames(root: Path, suffix: str,
                name_ok=lambda name: True) -> list[tuple[float | None, list[Path]]]:
    """VTK output files under root grouped per time step (all blocks of one
    output time together), sorted by the time in the filename. Slice outputs
    are excluded."""
    groups: dict[object, list[Path]] = {}
    for p in sorted(root.rglob(f"*{suffix}")):
        rel_parts = p.relative_to(root).parts
        if not p.is_file() or not name_ok(p.name.lower()) \
                or any(part.lower().startswith("slice") for part in rel_parts):
            continue
        m = TIME_IN_NAME.search(p.name)
        key = (p.parent.parent, float(m.group(1))) if m else p
        groups.setdefault(key, []).append(p)
    frames = [(k[1] if isinstance(k, tuple) else None, v) for k, v in groups.items()]
    return sorted(frames, key=lambda fr: (fr[0] is None, fr[0] or 0.0, str(fr[1][0])))


def frame_time(mesh, t_name: float | None) -> tuple[float | None, str]:
    """Output time from the 'TIME' field data if present, else from the filename."""
    for key in mesh.field_data.keys():
        if key.lower() == "time":
            return float(np.ravel(mesh.field_data[key])[0]), "field data 'TIME'"
    return t_name, "filename"


def analyze_vof_surface(pv, surf) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Per-structure volume, area and closedness of a VOF iso-surface.

    Degenerate triangles (M-Star pads its output with them) are dropped and
    duplicate points merged. Structures come from an M-Star structure/region
    ID array if present, else from connectivity. Volume uses the divergence
    theorem; a structure is closed if no edge belongs to only one triangle.
    Vectorized equivalent of extract_surface()/volume/area/n_open_edges per region.
    """
    empty = np.zeros(0)
    if surf.n_cells == 0:
        return empty, empty, empty.astype(bool), "none"
    if not surf.is_all_triangles:
        surf = surf.triangulate()
    f = surf.faces.reshape(-1, 4)[:, 1:]
    pts, merge = np.unique(np.asarray(surf.points), axis=0, return_inverse=True)
    f = merge.ravel()[f]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    if not keep.any():
        return empty, empty, empty.astype(bool), "none"
    mesh = pv.PolyData(pts, np.c_[np.full(keep.sum(), 3), f[keep]].ravel())

    id_name = None
    for data, per_point in ((surf.cell_data, False), (surf.point_data, True)):
        for name in data.keys():
            if "structure" in name.lower() or "region" in name.lower():
                id_name = name
                vals = np.asarray(data[name])
                if per_point:
                    vals = vals[surf.faces.reshape(-1, 4)[:, 1]]
                mesh.cell_data["sid"] = vals[keep]
                break
        if id_name:
            break

    if id_name:
        _, rid = np.unique(np.asarray(mesh.cell_data["sid"]).ravel(), return_inverse=True)
        source = f"structure ID array '{id_name}'"
    else:
        mesh = mesh.connectivity(extraction_mode="all")
        _, rid = np.unique(np.asarray(mesh.cell_data["RegionId"]), return_inverse=True)
        source = "connectivity"
    rid = rid.ravel()
    n = int(rid.max()) + 1

    F = mesh.faces.reshape(-1, 4)[:, 1:]
    P = np.asarray(mesh.points, float)
    p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    vol = np.abs(np.bincount(rid, np.einsum("ij,ij->i", p0, np.cross(p1, p2)) / 6.0, n))
    area = np.bincount(rid, 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1), n)

    edges = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1).astype(np.int64)
    key = edges[:, 0] * mesh.n_points + edges[:, 1]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    open_struct = np.zeros(n, bool)
    open_struct[np.tile(rid, 3)[cnt[inv.ravel()] == 1]] = True
    return vol, area, ~open_struct, source


VOF_METRICS = [  # (time-series CSV column, summary variable)
    ("n_drops", "Number of Drops n_drops [-]"),
    ("d32_m", "Sauter Diameter d32 (VOF) [m]"),
    ("d10_m", "Mean Drop Diameter d10 (VOF) [m]"),
    ("d43_m", "Volume-Mean Drop Diameter d43 (VOF) [m]"),
    ("d_max_m", "Largest Drop Diameter (VOF) [m]"),
    ("d32_excl_largest_m", "Sauter Diameter d32 excl. Largest (VOF) [m]"),
    ("F_disp", "Dispersed Fraction F_disp [-]"),
]


def vof_structures(case_dir: Path, min_voxels: float, voxel_size: float | None,
                   t_start: float, out_dir: Path | None = None) -> tuple[list[dict], dict]:
    """Drop size distribution and dispersed fraction from Volume VOF Surface .vtp files.

    Only closed structures (not clipped by walls/domain) are counted, and
    structures smaller than min_voxels * voxel_size^3 are dropped as debris.
    Writes vof_structures.csv and vof_dsd.csv to out_dir (default case_dir).
    Returns (summary rows, {"t", <metric columns>, "dsd"} for plotting).
    """
    pv = import_pyvista("VOF structure analysis")
    if pv is None:
        return [], {}
    frames = find_frames(case_dir, ".vtp", lambda n: "vof" in n)
    if not frames:
        warn(f"--dispersion: no VOF surface .vtp files (name containing 'VOF', "
             f"excluding Slice outputs) found under {case_dir}; skipping VOF structures")
        return [], {}
    log(f"VOF structures: {len(frames)} frame(s) under {case_dir}")
    if voxel_size:
        v_min = min_voxels * voxel_size ** 3
        log(f"  debris filter: V < {min_voxels:g} voxels = {v_min:.3g} m^3")
    else:
        v_min = 0.0
        warn("--dispersion: --voxel-size not given; keeping all VOF structures "
             "(no debris filter)")

    ts_rows, dsd_rows, dsd_plot = [], [], []
    sources, id_sources, n_open_frames = set(), set(), 0
    for t_name, paths in frames:
        meshes = [pv.read(p) for p in paths]
        t, src = frame_time(meshes[0], t_name)
        if t is None:
            warn(f"  {paths[0].name}: no time in field data or filename; skipped")
            continue
        sources.add(src)
        surf = meshes[0] if len(meshes) == 1 else pv.merge(meshes)
        vol, area, closed, id_src = analyze_vof_surface(pv, surf)
        id_sources.add(id_src)

        n_open = int((~closed).sum())
        n_open_frames += n_open > 0
        debris = closed & (vol < v_min)
        kept = closed & ~debris
        V, A = vol[kept], area[kept]
        d = (6.0 * V / np.pi) ** (1.0 / 3.0)
        rec = {"time_s": t, "n_regions": len(vol), "n_open": n_open,
               "n_debris": int(debris.sum()), "n_drops": int(kept.sum()),
               "V_total_m3": float(V.sum())}
        if V.size:
            big = int(np.argmax(V))
            rest = np.arange(V.size) != big
            rec.update({
                "d32_m": 6.0 * V.sum() / A.sum(),
                "d10_m": float(d.mean()),
                "d43_m": float((d ** 4).sum() / (d ** 3).sum()),
                "d_max_m": float(d.max()),
                "d32_excl_largest_m": (6.0 * V[rest].sum() / A[rest].sum()
                                       if rest.any() else np.nan),
                "F_disp": 1.0 - V[big] / V.sum(),
            })
            for i, gid in enumerate(np.flatnonzero(kept)):
                dsd_rows.append({"time_s": t, "region_id": int(gid), "d_eq_m": d[i],
                                 "volume_m3": V[i], "area_m2": A[i]})
                dsd_plot.append((t, d[i], V[i], i == big))
        else:
            rec.update({k: np.nan for k, _ in VOF_METRICS if k != "n_drops"})
        ts_rows.append(rec)
    if not ts_rows:
        return [], {}

    ts_rows.sort(key=lambda r: r["time_s"])
    log(f"  frame time taken from: {', '.join(sorted(sources))}")
    log(f"  structures identified by: {', '.join(sorted(id_sources))}")
    if n_open_frames:
        warn(f"  {n_open_frames} frame(s) contain open (clipped) structures, e.g. "
             "a bulk layer touching walls; these are excluded, so d32/F_disp "
             "refer to closed structures only")

    out_dir = out_dir or case_dir
    cols = ["time_s", "n_regions", "n_open", "n_debris", "n_drops"] \
        + [k for k, _ in VOF_METRICS if k != "n_drops"] + ["V_total_m3"]
    write_csv(ts_rows, out_dir / "vof_structures.csv", columns=cols, what="frames")
    write_csv(dsd_rows, out_dir / "vof_dsd.csv",
              columns=["time_s", "region_id", "d_eq_m", "volume_m3", "area_m2"],
              what="structures")

    t = np.array([r["time_s"] for r in ts_rows])
    series = {"t": t, "dsd": dsd_plot}
    rows = []
    for key, var in VOF_METRICS:
        series[key] = np.array([r.get(key, np.nan) for r in ts_rows], float)
        row = series_row("vof_structures.csv", var, t, series[key], t_start, "vof")
        if row:
            rows.append(row)
            log(f"  {var}: {row['mean']:.5g} \u00b1 {row['std']:.3g}")
    return rows, series


def phase_uniformity(case_dir: Path, n_zones_z: int, n_zones_r: int, t_start: float,
                     vertical_axis: str = "y", axis_center: list[float] | None = None,
                     out_dir: Path | None = None) -> tuple[list[dict], dict]:
    """Zonal CoV of the phase fraction and vertical holdup profile from .vti output.

    phi is coarse-grained into n_zones_z axial x n_zones_r radial zones before
    computing statistics (voxel phi is ~0/1). Writes phase_uniformity.csv and
    holdup_profile.csv. Returns (summary rows, {"t", "cov", "profile"}).
    """
    pv = import_pyvista("phase uniformity analysis")
    if pv is None:
        return [], {}
    frames = find_frames(case_dir, ".vti")
    if not frames:
        warn(f"--uniformity: no .vti volume files found under {case_dir}; skipping")
        return [], {}
    ax = "xyz".index(vertical_axis)
    h1, h2 = [a for a in range(3) if a != ax]
    log(f"phase uniformity: {len(frames)} frame(s) under {case_dir}, "
        f"vertical axis {vertical_axis}, {n_zones_z} x {n_zones_r} zones")

    phi_name = mask_name = None
    cov_rows, prof_rows, sources = [], [], set()
    for t_name, paths in frames:
        blocks = []  # (phi, vertical coord, radius) over fluid voxels
        t = src = None
        bounds = []
        for p in paths:
            reader = pv.get_reader(str(p))
            cell_names = list(reader.cell_array_names)
            names = cell_names + list(reader.point_array_names)
            if phi_name is None:
                cands = [n for n in names
                         if any(k in n.lower() for k in ("vof", "phase", "fraction"))]
                if not cands:
                    warn(f"--uniformity: no phase-fraction array (name containing "
                         f"'vof', 'phase' or 'fraction') in {p.name}; available: "
                         + ", ".join(names))
                    return [], {}
                if len(cands) > 1:
                    warn(f"--uniformity: ambiguous phase-fraction arrays "
                         f"{cands}; using '{cands[0]}'")
                phi_name = cands[0]
                masks = [n for n in names if any(k in n.lower() for k in ("solid", "flag"))]
                mask_name = masks[0] if masks else None
                log(f"  phase fraction: '{phi_name}'"
                    + (f", non-fluid mask: '{mask_name}' (nonzero = solid)" if mask_name else ""))
            if phi_name not in names:
                warn(f"  {p.name}: no '{phi_name}' array; skipped")
                continue
            reader.disable_all_cell_arrays()
            reader.disable_all_point_arrays()
            for n in (phi_name, mask_name):
                if n is None:
                    continue
                (reader.enable_cell_array if n in cell_names else reader.enable_point_array)(n)
            mesh = reader.read()
            t, src = frame_time(mesh, t_name)
            bounds.append(mesh.bounds)

            on_cells = phi_name in cell_names
            dims = np.array(mesh.dimensions) - (1 if on_cells else 0)
            off = 0.5 if on_cells else 0.0
            data = mesh.cell_data if on_cells else mesh.point_data
            shape = tuple(dims[::-1])  # VTK order: x fastest
            phi = np.asarray(data[phi_name], float).reshape(shape)
            fluid = np.isfinite(phi)
            if mask_name is not None and mask_name in data.keys():
                fluid &= np.asarray(data[mask_name]).reshape(shape) == 0
            c1d = [mesh.origin[a] + (np.arange(dims[a]) + off) * mesh.spacing[a]
                   for a in range(3)]
            zz, yy, xx = np.meshgrid(c1d[2], c1d[1], c1d[0], indexing="ij", sparse=True)
            coord = (xx, yy, zz)
            blocks.append((phi, coord, fluid, mesh.spacing[ax]))
        if not blocks or t is None:
            continue
        sources.add(src)

        b = np.array(bounds)
        if axis_center:
            c1, c2 = axis_center
        else:
            c1 = (b[:, 2 * h1].min() + b[:, 2 * h1 + 1].max()) / 2
            c2 = (b[:, 2 * h2].min() + b[:, 2 * h2 + 1].max()) / 2
        ph, vz, rr = [], [], []
        for phi, coord, fluid, dz in blocks:
            ph.append(phi[fluid])
            vz.append(np.broadcast_to(coord[ax], phi.shape)[fluid])
            rr.append(np.broadcast_to(np.sqrt((coord[h1] - c1) ** 2 + (coord[h2] - c2) ** 2),
                                      phi.shape)[fluid])
        ph, vz, rr = np.concatenate(ph), np.concatenate(vz), np.concatenate(rr)
        if ph.size == 0:
            warn(f"  t = {t:g} s: no fluid voxels; skipped")
            continue

        dz = blocks[0][3]
        z0, z1, r1 = vz.min() - dz / 2, vz.max() + dz / 2, rr.max()
        iz = np.minimum(((vz - z0) / (z1 - z0) * n_zones_z).astype(int), n_zones_z - 1)
        ir = np.minimum((rr / max(r1, 1e-30) * n_zones_r).astype(int), n_zones_r - 1)
        zone = iz * n_zones_r + ir
        nz = n_zones_z * n_zones_r
        w = np.bincount(zone, minlength=nz).astype(float)
        s = np.bincount(zone, ph, minlength=nz)
        occ = w > 0
        phi_zone = s[occ] / w[occ]
        mean = np.average(phi_zone, weights=w[occ])
        std = np.sqrt(np.average((phi_zone - mean) ** 2, weights=w[occ]))
        cov_rows.append({"time_s": t, "cov": std / mean if mean else np.nan,
                         "phi_mean": mean, "n_zones": int(occ.sum())})

        layer = np.rint((vz - vz.min()) / dz).astype(int)
        lw = np.bincount(layer)
        lsum = np.bincount(layer, ph)
        for k in np.flatnonzero(lw):
            prof_rows.append({"time_s": t, "z_m": vz.min() + k * dz,
                              "phi_mean": lsum[k] / lw[k]})
    if not cov_rows:
        return [], {}

    cov_rows.sort(key=lambda r: r["time_s"])
    log(f"  frame time taken from: {', '.join(sorted(sources))}")
    out_dir = out_dir or case_dir
    write_csv(cov_rows, out_dir / "phase_uniformity.csv",
              columns=["time_s", "cov", "phi_mean", "n_zones"], what="frames")
    write_csv(prof_rows, out_dir / "holdup_profile.csv",
              columns=["time_s", "z_m", "phi_mean"], what="rows")

    t = np.array([r["time_s"] for r in cov_rows])
    cov = np.array([r["cov"] for r in cov_rows], float)
    rows = []
    row = series_row("phase_uniformity.csv", "Zonal CoV of Phase Fraction [-]",
                     t, cov, t_start, "zonal")
    if row:
        rows.append(row)
        log(f"  {row['variable']}: {row['mean']:.4g} \u00b1 {row['std']:.3g}")
    return rows, {"t": t, "cov": cov, "profile": prof_rows}


def resolution_check(files: list[Path], t_start: float, voxel_size: float,
                     sigma: float | None, rho_c: float | None,
                     c_hinze: float = 0.725) -> list[dict]:
    """Hinze maximum stable drop size vs voxel size, from the steady mean
    energy dissipation rate in Fluid.txt."""
    if sigma is None or rho_c is None:
        warn("--dispersion: resolution check needs --sigma and --rho-continuous; skipped")
        return []
    fluid = [f for f in files if f.name.lower() == "fluid.txt"]
    table = load_stats_table(fluid[0]) if fluid else None
    if table is None:
        warn("--dispersion: no Fluid.txt for the resolution check; skipped")
        return []
    header, data = table
    cands = [j for j in range(1, len(header)) if "energy dissipation" in norm_name(header[j])]
    if not cands:
        warn("--dispersion: no energy dissipation column in Fluid.txt; "
             "resolution check skipped")
        return []
    # prefer the instantaneous per-unit-mass value
    j = min(cands, key=lambda j: (var_unit(header[j]).lower() != "w/kg",
                                  "time-avg" in header[j].lower()))
    row = series_row(fluid[0].name, header[j].strip(), data[:, 0], data[:, j],
                     t_start, "hinze")
    if row is None or row["mean"] <= 0:
        warn("--dispersion: no positive steady energy dissipation; resolution check skipped")
        return []
    eps = row["mean"]
    d_hinze = c_hinze * (sigma / rho_c) ** 0.6 * eps ** -0.4
    ratio = d_hinze / voxel_size
    log(f"resolution check: eps = {eps:.4g} ('{header[j].strip()}'), "
        f"d_max,Hinze = {d_hinze:.4g} m = {ratio:.3g} voxels")
    if ratio < 5:
        warn(f"Hinze maximum stable drop size is only {ratio:.3g} voxels (< 5): "
             "drop sizes are under-resolved, DSD/d32 results reflect the grid "
             "scale; macro-dispersion metrics (F_disp, holdup) remain meaningful")
    common = {"file": fluid[0].name, "steady_method": "hinze",
              "steady_start_s": t_start, "end_time_s": row["end_time_s"],
              "n_samples": row["n_samples"], "std": "", "min": "", "max": ""}
    return [{**common, "variable": "Hinze Max Stable Drop Size [m]", "unit": "m",
             "mean": d_hinze},
            {**common, "variable": "Hinze Drop Size / Voxel Size [-]", "unit": "-",
             "mean": ratio}]


# ---------------------------------------------------------------- plots

def plot_file(rel: str, header: list[str], data: np.ndarray, queries: list[str],
              t_start: float, out_html: Path,
              mix_vline: tuple[float, str] | None = None) -> int:
    """Write one interactive HTML of subplots for a stats file; returns subplot count."""
    import math

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    times = data[:, 0]
    traces = []  # (var_name, t, v, mean, std)
    for j in range(1, len(header)):
        name = header[j].strip()
        if queries and not any(q in norm_name(name) for q in queries):
            continue
        vals = data[:, j]
        ok = np.isfinite(times) & np.isfinite(vals)
        t, v = times[ok], vals[ok]
        if t.size == 0:
            continue
        steady = t >= t_start
        mean = float(v[steady].mean()) if steady.any() else np.nan
        std = float(v[steady].std(ddof=1)) if steady.sum() > 1 else 0.0
        traces.append((name, t, v, mean, std))
    if not traces:
        return 0

    n = len(traces)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[name for name, *_ in traces],
                        vertical_spacing=min(0.35 / nrows, 0.12),
                        horizontal_spacing=0.08)

    for i, (name, t, v, mean, std) in enumerate(traces):
        r, c = divmod(i, ncols)
        r, c = r + 1, c + 1
        fig.add_trace(go.Scatter(x=t, y=v, mode="lines", name=name,
                                 line=dict(width=1), showlegend=False,
                                 hovertemplate="t=%{x:.4g} s<br>%{y:.5g}<extra></extra>"),
                      row=r, col=c)
        fig.add_vline(x=t_start, line_dash="dash", line_color="green",
                      line_width=1, row=r, col=c)
        if mix_vline is not None:
            fig.add_vline(x=mix_vline[0], line_dash="dot", line_color="purple",
                          line_width=1, row=r, col=c)
        if np.isfinite(mean):
            fig.add_hrect(y0=mean - std, y1=mean + std, fillcolor="red",
                          opacity=0.1, line_width=0, row=r, col=c)
            fig.add_hline(y=mean, line_dash="dot", line_color="red", line_width=1,
                          annotation_text=f"{mean:.4g} \u00b1 {std:.3g}",
                          annotation_font_size=10, row=r, col=c)
        fig.update_xaxes(title_text="Time [s]", title_font_size=10, row=r, col=c)
        fig.update_yaxes(title_text=var_unit(name), title_font_size=10, row=r, col=c)

    fig.update_annotations(font_size=10)
    title = (f"{rel} (steady state: t \u2265 {t_start:.3g} s, green line; "
             f"red: steady mean \u00b1 std)")
    if mix_vline is not None:
        title += f"<br><sup>purple dotted line: {mix_vline[1]}</sup>"
    fig.update_layout(
        title=title,
        height=max(400, 300 * nrows), template="plotly_white", margin=dict(t=90))
    fig.write_html(str(out_html))
    return n


def make_plots(files: list[Path], stats_dir: Path, queries: list[str],
               t_start: float, out_dir: Path,
               mix_vlines: dict[Path, tuple[float, str]] | None = None) -> None:
    """One HTML per stats file; queries (if any) limit which variables are plotted."""
    qn = [q.strip().lower() for q in queries]
    out_dir.mkdir(parents=True, exist_ok=True)

    if qn:
        matched = {q: 0 for q in qn}
        for f in files:
            table = load_stats_table(f)
            if table is None:
                continue
            for h in table[0][1:]:
                for q in qn:
                    if q in norm_name(h):
                        matched[q] += 1
        for q, count in matched.items():
            if count == 0:
                warn(f"--plot '{q}' matched no variables")

    n_files = 0
    for f in files:
        table = load_stats_table(f)
        if table is None:
            continue
        header, data = table
        rel = str(f.relative_to(stats_dir))
        stem = re.sub(r"[^\w.-]+", "_", str(Path(rel).with_suffix("")))
        n = plot_file(rel, header, data, qn, t_start, out_dir / f"{stem}.html",
                      mix_vline=(mix_vlines or {}).get(f))
        if n:
            log(f"  {stem}.html: {n} subplot(s)")
            n_files += 1
    log(f"wrote {n_files} plot file(s) to {out_dir}")


def plot_dispersion(disp: dict, t_start: float, out_html: Path) -> None:
    """dispersion.html: time series of the dispersion metrics, pooled
    steady-state DSD and holdup-profile heatmap; panels without data are skipped."""
    import math

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    itf, vof, uni = disp.get("interface", {}), disp.get("vof", {}), disp.get("uniformity", {})
    panels = []  # (title, [traces], is_time_series, y-axis title)

    def line(t, y, name):
        return go.Scatter(x=t, y=y, mode="lines+markers", name=name, marker_size=4,
                          line_width=1,
                          hovertemplate="t=%{x:.4g} s<br>%{y:.5g}<extra>" + name + "</extra>")

    if "a" in itf:
        panels.append(("Specific interfacial area a(t)",
                       [line(itf["t"], itf["a"], "a (interface)")], True, "1/m"))
    d32 = []
    if "d32" in itf:
        d32.append(line(itf["t"], itf["d32"], "d32 (interface)"))
    if "d32_m" in vof:
        d32.append(line(vof["t"], vof["d32_m"], "d32 (VOF)"))
        d32.append(line(vof["t"], vof["d32_excl_largest_m"], "d32 excl. largest (VOF)"))
    if d32:
        panels.append(("Sauter diameter d32(t)", d32, True, "m"))
    if "F_disp" in vof:
        panels.append(("Dispersed fraction F_disp(t)",
                       [line(vof["t"], vof["F_disp"], "F_disp")], True, "-"))
        panels.append(("Number of drops n_drops(t)",
                       [line(vof["t"], vof["n_drops"], "n_drops")], True, "-"))
    if "cov" in uni:
        panels.append(("Zonal CoV of phase fraction (t)",
                       [line(uni["t"], uni["cov"], "CoV")], True, "-"))

    dsd = [(d, v) for t, d, v, largest in vof.get("dsd", []) if t >= t_start and not largest]
    if dsd:
        d, v = np.array(dsd).T
        edges = np.histogram_bin_edges(d, bins=min(40, max(5, int(np.sqrt(d.size)))))
        num, _ = np.histogram(d, edges)
        vw, _ = np.histogram(d, edges, weights=v)
        centers, width = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
        panels.append(("Steady-state DSD (largest structure excluded)", [
            go.Bar(x=centers, y=num / num.sum(), width=width, name="number-weighted",
                   opacity=0.6),
            go.Bar(x=centers, y=vw / vw.sum(), width=width, name="volume-weighted",
                   opacity=0.6),
        ], False, "fraction"))

    if uni.get("profile"):
        prof = uni["profile"]
        times = sorted({r["time_s"] for r in prof})
        zs = np.unique(np.round([r["z_m"] for r in prof], 9))
        grid = np.full((zs.size, len(times)), np.nan)
        ti = {t: i for i, t in enumerate(times)}
        for r in prof:
            grid[np.searchsorted(zs, round(r["z_m"], 9)), ti[r["time_s"]]] = r["phi_mean"]
        panels.append(("Holdup profile \u03c6(z, t)", [
            go.Heatmap(x=times, y=zs, z=grid, colorscale="Viridis",
                       colorbar=dict(title="\u03c6", len=0.3, y=0.15),
                       hovertemplate="t=%{x:.4g} s<br>z=%{y:.4g} m<br>\u03c6=%{z:.4g}<extra></extra>")
        ], True, "z [m]"))

    if not panels:
        warn("--plot: no dispersion data to plot")
        return
    ncols = 2 if len(panels) > 1 else 1
    nrows = math.ceil(len(panels) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[p[0] for p in panels],
                        vertical_spacing=min(0.35 / nrows, 0.12), horizontal_spacing=0.1)
    for i, (title, traces, timeseries, ytitle) in enumerate(panels):
        r, c = divmod(i, ncols)
        r, c = r + 1, c + 1
        for tr in traces:
            fig.add_trace(tr, row=r, col=c)
        if timeseries:
            fig.add_vline(x=t_start, line_dash="dash", line_color="green",
                          line_width=1, row=r, col=c)
            fig.update_xaxes(title_text="Time [s]", title_font_size=10, row=r, col=c)
        else:
            fig.update_xaxes(title_text="d_eq [m]", title_font_size=10, row=r, col=c)
        fig.update_yaxes(title_text=ytitle, title_font_size=10, row=r, col=c)
    fig.update_annotations(font_size=10)
    fig.update_layout(
        title=f"Phase dispersion (steady state: t \u2265 {t_start:.3g} s, green line)",
        barmode="overlay", height=max(400, 320 * nrows), template="plotly_white",
        margin=dict(t=90))
    fig.write_html(str(out_html))
    log(f"wrote {out_html} ({len(panels)} panel(s))")


# ---------------------------------------------------------------- main

def process_case(case: Path, stats_dir: Path, args,
                 output: Path | None = None, plot_dir: Path | None = None) -> list[dict]:
    """Summarize one case; write its CSV (and plots); return its rows ([] on failure)."""
    files = find_stats_files(stats_dir)
    if not files:
        warn(f"no .txt stats files found in {stats_dir}")
        return []
    log(f"stats folder: {stats_dir} ({len(files)} files)")

    if args.time is not None:
        t_start, method = args.time, "user"
        log(f"user-defined steady-state start: t >= {t_start} s")
    else:
        t_start = steady_from_fluid_velocity(files, args.window, args.tol)
        method = "auto"
        if t_start is None:
            warn(f"{case}: could not detect a mean-velocity plateau; "
                 "specify --time instead")
            return []
        log(f"steady state detected at t = {t_start:.3f} s "
            f"(mean fluid velocity plateau, window={args.window} s, tol={args.tol})")

    rows = []
    for f in files:
        rel = str(f.relative_to(stats_dir))
        file_rows = summarize_file(f, rel, t_start, method)
        if file_rows:
            log(f"  {rel}: {len(file_rows)} variables")
            rows.extend(file_rows)
        else:
            warn(f"  {rel}: skipped (not a time-course stats table, or no data after t_start)")
    if not rows:
        warn(f"{case}: no variables summarized")
        return []

    rows_mix, mix_vlines = scalar_mixing(files, stats_dir, args.mix_rsd)
    rows.extend(rows_mix)

    base = case if case.is_dir() else case.parent
    disp: dict = {}
    if args.dispersion:
        rows_itf, disp["interface"] = interface_metrics(
            files, stats_dir, t_start, args.dispersed_fraction, args.liquid_volume)
        rows.extend(rows_itf)
        vof_root = base
        if args.vof_dir is not None:
            vof_root = args.vof_dir if args.vof_dir.is_absolute() or not args.batch \
                else base / args.vof_dir
        rows_vof, disp["vof"] = vof_structures(vof_root, args.min_voxels, args.voxel_size,
                                               t_start, out_dir=base)
        rows.extend(rows_vof)
        if args.voxel_size:
            rows.extend(resolution_check(files, t_start, args.voxel_size, args.sigma,
                                         args.rho_continuous, args.hinze_c))
    if args.uniformity:
        rows_uni, disp["uniformity"] = phase_uniformity(
            base, args.zones_z, args.zones_r, t_start, args.vertical_axis,
            args.axis_center, out_dir=base)
        rows.extend(rows_uni)

    write_csv(rows, output or base / "stats_summary.csv")

    if args.plot is not None:
        make_plots(files, stats_dir, args.plot, t_start,
                   plot_dir or base / "stats_plots", mix_vlines=mix_vlines)
        if disp:
            plot_dispersion(disp, t_start, (plot_dir or base / "stats_plots") / "dispersion.html")
    return rows


def find_batch_cases(parent: Path) -> list[tuple[Path, Path]]:
    """Return (case dir, stats dir) for each immediate subfolder with stats files."""
    cases = []
    for sub in sorted(p for p in parent.iterdir() if p.is_dir()):
        stats_dir = locate_stats_dir(sub)
        if stats_dir is not None:
            cases.append((sub, stats_dir))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("case", type=Path,
                    help="M-Star case directory, or the Stats directory itself; "
                         "with --batch, a parent directory of case subfolders")
    ap.add_argument("--batch", action="store_true",
                    help="treat CASE as a parent directory and process every "
                         "subfolder containing stats; writes one CSV per case "
                         "plus a combined batch CSV")
    ap.add_argument("--time", type=float, default=None,
                    help="user-defined steady-state start time [s]; "
                         "overrides automatic detection")
    ap.add_argument("--window", type=float, default=1.0,
                    help="steady-state detection window [s]")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="steady-state tolerance on windowed-mean drift")
    ap.add_argument("--mix-rsd", type=float, default=5.0, metavar="PCT",
                    help="RSD threshold [%%] for the mixed-time of scalar "
                         "tracers (5%% RSD = 95%% mixed)")
    ap.add_argument("--output", type=Path, default=None,
                    help="output CSV path (default: <case>/stats_summary.csv; "
                         "with --batch, the batch CSV, default "
                         "<parent>/batch_summary.csv)")
    ap.add_argument("--means-vars", nargs="*", metavar="VAR", default=None,
                    help="variable names (fuzzy) to include as columns in the "
                         "wide means CSV (default: energy dissipation, "
                         "velocity, vorticity, mixed time, shear, power "
                         "number, fluid volume, viscosity, density, rotation "
                         "speed, angular velocity, interfacial area, d32, "
                         "dispersed fraction, n_drops, cov)")
    ap.add_argument("--plot", nargs="*", metavar="VAR", default=None,
                    help="write one interactive HTML of plots per stats file; "
                         "give variable names (fuzzy) to limit which are plotted, "
                         "or no names to plot all variables")
    ap.add_argument("--plot-dir", type=Path, default=None,
                    help="plot output directory (default: <case>/stats_plots; "
                         "ignored with --batch)")

    dg = ap.add_argument_group(
        "phase dispersion (Immiscible Two-Fluid; VTK parts need pyvista)",
        "example: mstar_stats_summary.py case --dispersion --dispersed-fraction 0.2 "
        "--voxel-size 5e-4 --sigma 0.05 --rho-continuous 998 --plot")
    dg.add_argument("--dispersion", action="store_true",
                    help="interfacial-area metrics from the interface stats file, "
                         "drop size distribution / dispersed fraction from the "
                         "Volume VOF Surface .vtp output, and (with --voxel-size, "
                         "--sigma, --rho-continuous) a Hinze resolution check")
    dg.add_argument("--dispersed-fraction", type=float, default=None, metavar="PHI",
                    help="dispersed-phase volume fraction, for the area-based "
                         "d32 = 6*PHI/a")
    dg.add_argument("--liquid-volume", type=float, default=None, metavar="M3",
                    help="liquid volume [m^3] for a = A_int/V_liq "
                         "(default: 'Fluid Volume' from Fluid.txt)")
    dg.add_argument("--vof-dir", type=Path, default=None,
                    help="directory searched recursively for VOF surface .vtp "
                         "files (default: the case directory; relative to each "
                         "case with --batch)")
    dg.add_argument("--voxel-size", type=float, default=None, metavar="M",
                    help="lattice spacing [m] for the debris filter and the "
                         "resolution check")
    dg.add_argument("--min-voxels", type=float, default=8,
                    help="VOF structures smaller than this many voxels are "
                         "dropped as numerical debris")
    dg.add_argument("--sigma", type=float, default=None,
                    help="interfacial tension [N/m] for the Hinze resolution check")
    dg.add_argument("--rho-continuous", type=float, default=None, metavar="RHO",
                    help="continuous-phase density [kg/m^3] for the Hinze "
                         "resolution check")
    dg.add_argument("--hinze-c", type=float, default=0.725,
                    help="Hinze constant C in d_max = C*(sigma/rho_c)^0.6*eps^-0.4")
    dg.add_argument("--uniformity", action="store_true",
                    help="zonal CoV and vertical holdup profile of the phase "
                         "fraction from the 3D Volume .vti output")
    dg.add_argument("--zones-z", type=int, default=10,
                    help="number of axial zones for the zonal CoV")
    dg.add_argument("--zones-r", type=int, default=3,
                    help="number of radial zones for the zonal CoV")
    dg.add_argument("--vertical-axis", choices="xyz", default="y",
                    help="vessel axis / vertical direction (M-Star is y-up)")
    dg.add_argument("--axis-center", type=float, nargs=2, default=None,
                    metavar=("A", "B"),
                    help="vessel-axis position in the two horizontal coordinates "
                         "(x z for a y axis; default: domain center)")
    args = ap.parse_args(argv)

    if args.dispersed_fraction is not None and not 0 < args.dispersed_fraction < 1:
        fail("--dispersed-fraction must be between 0 and 1")
    if args.zones_z < 1 or args.zones_r < 1:
        fail("--zones-z and --zones-r must be >= 1")

    if args.batch:
        if not args.case.is_dir():
            fail(f"--batch requires a directory: {args.case}")
        cases = find_batch_cases(args.case)
        if not cases:
            fail(f"no case subfolders with stats found under {args.case}")
        log(f"batch mode: {len(cases)} case(s) under {args.case}")

        batch_rows = []
        for case, stats_dir in cases:
            log(f"=== case: {case.name} ===")
            rows = process_case(case, stats_dir, args)
            for r in rows:
                batch_rows.append({"case": case.name, **r})
        if not batch_rows:
            fail("no variables summarized in any case")
        out = args.output or args.case / "batch_summary.csv"
        write_csv(batch_rows, out, columns=["case"] + COLUMNS)
        write_wide_csv(batch_rows, out.with_name(f"{out.stem}_means{out.suffix}"),
                       args.means_vars or DEFAULT_MEANS_VARS,
                       warn_unmatched=args.means_vars is not None,
                       exclude=None if args.means_vars else DEFAULT_MEANS_EXCLUDE)
    else:
        stats_dir = find_stats_dir(args.case)
        base = args.case if args.case.is_dir() else args.case.parent
        out = args.output or base / "stats_summary.csv"
        rows = process_case(args.case, stats_dir, args,
                            output=out, plot_dir=args.plot_dir)
        if not rows:
            fail("no variables summarized")
        write_wide_csv([{"case": args.case.name, **r} for r in rows],
                       out.with_name(f"{out.stem}_means{out.suffix}"),
                       args.means_vars or DEFAULT_MEANS_VARS,
                       warn_unmatched=args.means_vars is not None,
                       exclude=None if args.means_vars else DEFAULT_MEANS_EXCLUDE)
    log("done")


if __name__ == "__main__":
    main()
