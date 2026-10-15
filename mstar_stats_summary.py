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
                d32 = 6*phi/a (only if closed drops carry >= 50% of the
                interface) from the interface stats file, and the dispersion
                kinetics of a(t): a0, a_ss, interface enhancement E = a_ss/a0,
                t90/t95, time constant tau, plateau CoV and integral of a dt;
                per-fluid drop
                statistics from the Volume VOF Surface .vtp output
                (vof_structures.csv, vof_dsd.csv): closed drops of at least
                --min-d-dx voxels are assigned
                to the fluid inside them (from the .vti volume fraction),
                giving n_drops, d32, d32/dx, d10/d43 and F_disp = V_drops/V_fluid
                per fluid, plus the interface coverage A_drops/A_int; skipped
                for free-surface cases. Needs the voxel size (--voxel-size or
                <dx> in input.xml); with --sigma and --rho-continuous also a
                Hinze drop-size resolution check (with optional holdup and
                local-eps corrections --k-phi, --eps-ratio, --eps-file).
  --uniformity  zonal coefficient of variation of the phase fraction and the
                vertical holdup profile from the 3D Volume .vti output
                (phase_uniformity.csv, holdup_profile.csv); for two fluids
                also the segregation index S at each --zone-scales scale, the
                homogeneity time t_S, and the continuous phase from the
                largest connected region of each fluid (phase_continuity.csv,
                needs scipy).
VTK-based metrics are averaged from t_disp = max(flow steady state, a(t)
plateau start), or --disp-time.
With --plot, the dispersion results are also plotted to stats_plots/dispersion.html.
The --dispersion / --uniformity metrics are also written on their own to
stats_summary_{dispersion,uniformity}.csv and, as one-row-per-case means with
bar plots, *_{dispersion,uniformity}_means.csv/.html (batch: batch_summary_*).
Simulation settings (dx, LX/LY/LZ, dt, Courant number, fluid volume, impeller
speeds, fluid properties) from input.xml and Fluid.txt go to *_settings.csv.

Example:
    python mstar_stats_summary.py test              # auto-detect steady state
    python mstar_stats_summary.py test --time 5.0   # steady state = t >= 5 s
    python mstar_stats_summary.py test --plot       # plot all variables
    python mstar_stats_summary.py test --plot "mean velocity" "power number"
    python mstar_stats_summary.py runs --batch      # process every case in runs/
    python mstar_stats_summary.py case --dispersion --dispersed-fraction 0.2 \
        --voxel-size 5e-4 --sigma 0.05 --rho-continuous 998 --plot
    python mstar_stats_summary.py case --uniformity --zones-z 10 --zones-r 3
    python mstar_stats_summary.py runs --batch --dispersion --uniformity \
        --zone-scales 5x2 10x3 20x5 --sigma 0.001 --rho-continuous 998 --ref-case base

Cases:
- /data2/mstar/tak360/step8

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

def is_hidden(path: Path, root: Path) -> bool:
    """True if any folder of path below root starts with '.' (e.g. .archive)."""
    return any(part.startswith(".") for part in path.relative_to(root).parts[:-1]) \
        or (path.is_dir() and path.name.startswith(".") and path != root)


def locate_stats_dir(root: Path) -> Path | None:
    """Accept a case dir, a dir containing Stats/, or the Stats dir itself."""
    if root.is_dir():
        if root.name.lower() == "stats":
            return root
        hits = sorted(p for p in root.rglob("Stats") if p.is_dir() and not is_hidden(p, root))
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
    return sorted(f for f in stats_dir.rglob("*.txt")
                  if f.is_file() and not is_hidden(f, stats_dir))


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
    "coverage", "hinze", "interface enhancement", "dispersion time",
    "time-integrated", "segregation", "homogeneity time", "continuous phase",
]
# with the default list, keep only mean values: drop LB and min/max variants
DEFAULT_MEANS_EXCLUDE = r"\b(lb|min|max)\b"


def write_wide_csv(batch_rows: list[dict], out_path: Path, queries: list[str],
                   warn_unmatched: bool = True, exclude: str | None = None,
                   ref_case: str | None = None) -> None:
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

    plot_means_bars(cols, data, out_path.with_suffix(".html"), ref_case)


def plot_means_bars(cols: list[str], data: dict[str, dict[str, object]],
                    out_html: Path, ref_case: str | None = None) -> None:
    """Bar plots of the means table: absolute values and % difference vs the
    reference case (ref_case, exact or unique substring; default the first
    case), one row of subplots per variable."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    cases = list(data)
    ref = cases[0]
    if ref_case is not None:
        hits = [c for c in cases if c == ref_case] \
            or [c for c in cases if ref_case.lower() in c.lower()]
        if len(hits) == 1:
            ref = hits[0]
        else:
            warn(f"--ref-case '{ref_case}' matched {len(hits)} cases; using '{ref}'")
    i_ref = cases.index(ref)
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
        ref_v = ys[i_ref]
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


def value_row(file: str, variable: str, value, method: str, t_start="", end="",
              n="") -> dict:
    """Summary row holding a single derived value (no std/min/max)."""
    return {"file": file, "variable": variable, "unit": var_unit(variable),
            "steady_method": method, "steady_start_s": t_start, "end_time_s": end,
            "n_samples": n, "mean": value, "std": "", "min": "", "max": ""}


def approach_kinetics(t, y, window: float, tol: float,
                      levels=(0.90, 0.95)) -> dict | None:
    """Approach of y(t) from its first sample y0 to its plateau y_ss.

    Returns y0, plateau start t_ss (detect_steady_start), y_ss = mean over the
    plateau, plateau_cov = std/y_ss there, t<level> = first time
    frac = (y - y0)/(y_ss - y0) reaches each level (linear interpolation) and
    tau from a log-linear fit of ln(1 - frac) vs t over 0.05 < frac < 0.95.
    Plateau-dependent values are NaN if y never plateaus; None if < 2 samples.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size < 2:
        return None
    keys = [f"t{100 * lev:g}" for lev in levels]
    out = {"y0": float(y[0]), "t_ss": np.nan, "y_ss": np.nan, "plateau_cov": np.nan,
           "tau": np.nan, **{k: np.nan for k in keys}}
    t_ss = detect_steady_start(t, y, window, tol)
    if t_ss is None:
        return out
    plateau = t >= t_ss
    y_ss = float(y[plateau].mean())
    out.update(t_ss=t_ss, y_ss=y_ss)
    if plateau.sum() > 1 and y_ss:
        out["plateau_cov"] = float(y[plateau].std(ddof=1) / abs(y_ss))
    if y_ss == out["y0"]:
        return out
    frac = (y - out["y0"]) / (y_ss - out["y0"])
    for key, lev in zip(keys, levels):
        hit = np.flatnonzero(frac >= lev)
        if hit.size:
            i = hit[0]
            out[key] = float(t[0] if i == 0 else t[i - 1] + (lev - frac[i - 1])
                             / (frac[i] - frac[i - 1]) * (t[i] - t[i - 1]))
    hit = np.flatnonzero(frac >= 0.95)
    rise = np.arange(t.size) < (hit[0] if hit.size else t.size)
    sel = rise & (frac > 0.05) & (frac < 0.95)
    if sel.sum() >= 3:
        slope = np.polyfit(t[sel], np.log(1.0 - frac[sel]), 1)[0]
        if slope < 0:
            out["tau"] = float(-1.0 / slope)
    return out


def dispersion_kinetics(t, a, window: float, tol: float, levels=(0.90, 0.95)) -> dict:
    """Dispersion kinetics of the specific interfacial area a(t): a0 (flat
    interface at impeller start), plateau start t_ss_a, a_ss, enhancement
    E = a_ss/a0, t90/t95, time constant tau, plateau CoV and int_a = integral
    of a dt from t = 0. Empty dict if a(t) has < 2 samples."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    ok = np.isfinite(t) & np.isfinite(a)
    t, a = t[ok], a[ok]
    k = approach_kinetics(t, a, window, tol, levels)
    if k is None:
        return {}
    a0 = k["y0"]
    if t[0] > 0:  # a is the flat-interface a0 until the first sample
        t, a = np.r_[0.0, t], np.r_[a0, a]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    out = {"a0": a0, "t_ss_a": k["t_ss"], "a_ss": k["y_ss"],
           "E": k["y_ss"] / a0 if a0 > 0 else np.nan, "tau": k["tau"],
           "plateau_cov": k["plateau_cov"], "int_a": float(trapz(a, t))}
    out.update({f"t{100 * lev:g}": k[f"t{100 * lev:g}"] for lev in levels})
    return out


KINETICS_VARS = [  # (dispersion_kinetics key, summary variable)
    ("a0", "Initial Specific Interfacial Area a0 [1/m]"),
    ("a_ss", "Plateau Specific Interfacial Area a_ss [1/m]"),
    ("t_ss_a", "Interfacial Area Plateau Start t_ss_a [s]"),
    ("E", "Interface Enhancement E = a_ss/a0 [-]"),
    ("t90", "Dispersion Time t90 [s]"),
    ("t95", "Dispersion Time t95 [s]"),
    ("tau", "Dispersion Time Constant tau [s]"),
    ("plateau_cov", "Interfacial Area Plateau CoV [-]"),
    ("int_a", "Time-Integrated Specific Interfacial Area [s/m]"),
]


def interface_metrics(files: list[Path], stats_dir: Path, t_start: float,
                      phi: float | None, v_liq: float | None, window: float = 1.0,
                      tol: float = 0.02, vessel_diameter: float | None = None,
                      area_column: str | None = None) -> tuple[list[dict], dict]:
    """Specific interfacial area a = A_int/V_liq, area-based d32 = 6*phi/a and
    the dispersion kinetics of a(t) (dispersion_kinetics()).

    Returns (summary rows, {"t", "a", "area"[, "d32"], "kinetics"} time series).
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
    if area_column:
        q = area_column.strip().lower()
        cols = [j for j in range(1, len(header)) if q in normed[j]]
        if not cols:
            warn(f"--dispersion: --area-column '{q}' matched no column in {rel}; "
                 "available: " + ", ".join(h.strip() for h in header[1:]))
            return [], {}
    elif len(cols) > 1:
        warn(f"--dispersion: several area columns in {rel}: "
             + ", ".join(f"'{header[j].strip()}'" for j in cols)
             + f"; using '{header[cols[0]].strip()}' (choose with --area-column)")
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
    series = {"t": t, "a": a, "area": area}
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

    kin = dispersion_kinetics(t, a, window, tol)
    series["kinetics"] = kin
    if not kin:
        return rows, series
    tf = t[np.isfinite(t) & np.isfinite(a)]
    if tf.size > 1 and tf[0] > np.median(np.diff(tf)):
        warn(f"--dispersion: first a(t) sample at t = {tf[0]:g} s, more than one "
             "output interval after 0; a0 may not be the flat interface")
    if not np.isfinite(kin["t_ss_a"]):
        warn("--dispersion: a(t) never plateaus: dispersion still developing; "
             "extend the run")
    if vessel_diameter:
        v0 = vol[np.isfinite(t) & np.isfinite(a)][0]
        a_flat = np.pi * vessel_diameter ** 2 / 4.0 / v0
        dev = kin["a0"] / a_flat - 1.0
        (warn if abs(dev) > 0.25 else log)(
            f"  a0 = {kin['a0']:.4g} 1/m vs flat-interface estimate "
            f"pi*D^2/4/V_liq = {a_flat:.4g} 1/m ({100 * dev:+.0f}%)")
    t_ss = kin["t_ss_a"] if np.isfinite(kin["t_ss_a"]) else ""
    for key, var in KINETICS_VARS:
        v = kin.get(key, np.nan)
        if np.isfinite(v):
            rows.append(value_row(rel, var, float(v), "a(t) kinetics", t_ss,
                                  float(tf[-1]), int(tf.size)))
    log("  a(t) kinetics: " + ", ".join(f"{k} = {kin[k]:.4g}" for k, _ in KINETICS_VARS))
    return rows, series


TIME_IN_NAME = re.compile(r"\.([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\.vt[ip]$")


def find_frames(root: Path, suffix: str,
                name_ok=lambda name: True) -> list[tuple[float | None, list[Path]]]:
    """VTK output files under root grouped per time step (all blocks of one
    output time together), sorted by the time in the filename. Slice outputs
    and hidden folders are excluded."""
    groups: dict[object, list[Path]] = {}
    for p in sorted(root.rglob(f"*{suffix}")):
        rel_parts = p.relative_to(root).parts
        if not p.is_file() or not name_ok(p.name.lower()) or is_hidden(p, root) \
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


def fluid_volumes(files: list[Path]) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    """Per-fluid volumes from the 'Fluid N (name) Volume' columns of Fluid.txt,
    as [(N, name, t, V)] sorted by N; empty for free-surface/single-fluid cases."""
    fluid = [f for f in files if f.name.lower() == "fluid.txt"]
    table = load_stats_table(fluid[0]) if fluid else None
    if table is None:
        return []
    header, data = table
    out = []
    for j, h in enumerate(header):
        m = re.match(r"\s*fluid\s*(\d+)\s*\((.*?)\)\s*volume\s*(\[[^\]]*\])?\s*$", h, re.I)
        if m:
            ok = np.isfinite(data[:, 0]) & np.isfinite(data[:, j])
            out.append((int(m.group(1)), m.group(2).strip() or f"Fluid {m.group(1)}",
                        data[ok, 0], data[ok, j]))
    return sorted(out, key=lambda x: x[0])


def find_input_xml(base: Path) -> Path | None:
    """The M-Star input.xml in or up to two levels above the case dir."""
    for d in [base, *list(base.parents)[:2]]:
        p = d / "input.xml"
        if p.is_file():
            return p
    return None


def input_voxel_size(base: Path) -> float | None:
    """Lattice spacing <dx> from the M-Star input.xml in or above the case dir."""
    p = find_input_xml(base)
    if p:
        m = re.search(r"<dx>\s*([-+0-9.eE]+)\s*</dx>", p.read_text(errors="replace"))
        if m:
            return float(m.group(1))
    return None


def case_settings(case: Path, stats_dir: Path) -> dict:
    """Simulation settings of one case from input.xml (dx, LX/LY/LZ, dt, Courant
    number, impeller speeds, per-fluid density/viscosity, surface tension) and
    Fluid.txt (initial fluid volume, max simulated Courant number)."""
    import xml.etree.ElementTree as ET

    def num(el, tag):
        txt = el.findtext(tag) if el is not None else None
        if txt is None or not txt.strip():
            return None
        try:
            return float(txt)
        except ValueError:  # e.g. an expression
            return txt.strip()

    base = case if case.is_dir() else case.parent
    s: dict = {"case": case.name}
    dx = dt = None
    p = find_input_xml(base)
    root = None
    if p is None:
        warn(f"{case.name}: no input.xml found; settings limited to Fluid.txt")
    else:
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError as e:
            warn(f"{p}: unreadable ({e}); settings limited to Fluid.txt")
    if root is not None:
        s["input_xml"] = str(p)
        sys_el = root.find("system")
        dx, dt = num(sys_el, "dx"), num(sys_el, "dt")
        s["dx [m]"] = dx
        for a in "XYZ":
            length = num(sys_el, f"domain{a}")
            if isinstance(length, float) and isinstance(dx, float) and dx > 0:
                s[f"L{a} [-]"] = int(round(length / dx))
        s["dt [s]"] = dt
        # M-Star sets dt = uMax*dx/U_ref, so uMax is the Courant number at U_ref
        s["Courant number (input uMax) [-]"] = num(sys_el, "uMax")
        for imp in root.findall("impeller"):
            name = (imp.findtext("name") or "impeller").strip()
            s[f"Impeller speed ({name}) [rpm]"] = num(imp, "freq")
        fm = root.find("fluidModel")
        if fm is not None:
            for i, rh in enumerate(fm.findall("rheology"), 1):
                s[f"Fluid {i} name"] = (fm.findtext(f"fluid{i}Name") or "").strip()
                rho, nu = num(rh, "density"), num(rh, "nu")
                s[f"Fluid {i} density [kg/m^3]"] = rho
                s[f"Fluid {i} kinematic viscosity [m^2/s]"] = nu
                if isinstance(rho, float) and isinstance(nu, float):
                    s[f"Fluid {i} dynamic viscosity [Pa s]"] = rho * nu
            s["Surface tension [N/m]"] = num(fm, "surfaceTension")

    fluid = [f for f in find_stats_files(stats_dir) if f.name.lower() == "fluid.txt"]
    table = load_stats_table(fluid[0]) if fluid else None
    if table is not None:
        header, data = table
        normed = [norm_name(h) for h in header]
        if "fluid volume" in normed:
            v = data[:, normed.index("fluid volume")]
            v = v[np.isfinite(v)]
            if v.size:
                s["Fluid volume (total, initial) [m^3]"] = float(v[0])
        if "max velocity" in normed and isinstance(dx, float) and isinstance(dt, float):
            u = data[:, normed.index("max velocity")]
            if np.isfinite(u).any():
                s["Max Courant number (simulated, max over run) [-]"] = \
                    float(np.nanmax(u)) * dt / dx
    return {k: v for k, v in s.items() if v is not None}


def write_settings_csv(settings: list[dict], out_path: Path) -> None:
    cols: list[str] = []
    for s in settings:
        cols += [k for k in s if k not in cols]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(settings)
    log(f"wrote {out_path} ({len(settings)} cases x {len(cols) - 1} settings)")


def vti_sampler(pv, paths: list[Path], fluid_idx: int):
    """Point sampler of the 'Fluid <fluid_idx> ... Volume Fraction' cell array of
    the (multi-block) .vti volume output; None if the array is missing."""
    pat = re.compile(rf"fluid\s*{fluid_idx}\b.*volume fraction", re.I)
    blocks = []
    for p in paths:
        reader = pv.get_reader(str(p))
        name = next((n for n in reader.cell_array_names if pat.search(n)), None)
        if name is None:
            continue
        reader.disable_all_cell_arrays()
        reader.disable_all_point_arrays()
        reader.enable_cell_array(name)
        mesh = reader.read()
        dims = np.array(mesh.dimensions) - 1
        blocks.append((np.array(mesh.origin), np.array(mesh.spacing), dims,
                       np.asarray(mesh.cell_data[name], float).reshape(dims[::-1])))
    if not blocks:
        return None

    def sample(pts: np.ndarray) -> np.ndarray:
        out = np.full(len(pts), np.nan)
        for origin, spacing, dims, arr in blocks:
            ijk = np.floor((pts - origin) / spacing).astype(int)
            ok = np.all((ijk >= 0) & (ijk < dims), axis=1) & np.isnan(out)
            i, j, k = ijk[ok].T
            out[ok] = arr[k, j, i]
        return out
    return sample


def analyze_vof_surface(pv, surf, sampler=None, voxel_size: float | None = None,
                        n_samples: int = 25) -> dict:
    """Per-structure volume, area, closedness and inside fluid of a VOF iso-surface.

    Degenerate triangles (M-Star pads its output with them) are dropped and
    duplicate points merged (within 1e-3 voxels, so drops crossing block
    seams stay closed). Structures come from an M-Star structure/region
    ID array if present, else from connectivity. Volume uses the divergence
    theorem; a structure is closed if no edge belongs to only one triangle.

    "inside" is the fluid-1 volume fraction inside each closed structure
    (NaN for open ones): sampler(points) at up to n_samples points just inside
    its surface ("inside_surface"), replaced for drops < 4 voxels across by the
    sample at the volume-weighted centroid ("inside_centroid"), where the
    surface-offset points can land in interface cells. Without a sampler it is
    inferred from the triangle orientation (M-Star orients the iso-surface
    normals toward fluid 2).
    Returns {"vol", "area", "closed", "inside", "inside_surface",
    "inside_centroid", "signed", "source"}.
    """
    empty = np.zeros(0)
    none = {"vol": empty, "area": empty, "closed": empty.astype(bool),
            "inside": empty, "inside_surface": empty, "inside_centroid": empty,
            "signed": empty, "source": "none"}
    if surf.n_cells == 0:
        return none
    if not surf.is_all_triangles:
        surf = surf.triangulate()
    f = surf.faces.reshape(-1, 4)[:, 1:]
    raw = np.asarray(surf.points, float)
    if voxel_size:
        q = np.round(raw / (1e-3 * voxel_size)).astype(np.int64)
        _, first, merge = np.unique(q, axis=0, return_index=True, return_inverse=True)
        pts = raw[first]
    else:
        pts, merge = np.unique(raw, axis=0, return_inverse=True)
    f = merge.ravel()[f]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    if not keep.any():
        return none
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
    tet = np.einsum("ij,ij->i", p0, np.cross(p1, p2)) / 6.0
    signed = np.bincount(rid, tet, n)
    vol = np.abs(signed)
    nrm = np.cross(p1 - p0, p2 - p0)
    area = np.bincount(rid, 0.5 * np.linalg.norm(nrm, axis=1), n)

    edges = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1).astype(np.int64)
    key = edges[:, 0] * mesh.n_points + edges[:, 1]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    open_struct = np.zeros(n, bool)
    open_struct[np.tile(rid, 3)[cnt[inv.ravel()] == 1]] = True
    closed = ~open_struct

    # positive signed volume = outward normals = fluid 2 outside = fluid-1 drop
    inside = np.where(closed, (signed > 0).astype(float), np.nan)
    if sampler is not None and closed.any():
        fc = np.flatnonzero(closed[rid])
        order = fc[np.argsort(rid[fc], kind="stable")]
        r = rid[order]
        pick = order[np.arange(r.size) - np.searchsorted(r, r) < n_samples]
        rp = rid[pick]
        unit = nrm[pick] / np.maximum(np.linalg.norm(nrm[pick], axis=1, keepdims=True), 1e-300)
        outward = np.sign(signed[rp])[:, None] * unit
        depth = 0.25 * (6.0 * vol[rp] / np.pi) ** (1.0 / 3.0)
        if voxel_size:
            depth = np.minimum(depth, voxel_size)
        s = sampler((p0[pick] + p1[pick] + p2[pick]) / 3.0 - outward * depth[:, None])
        ok = np.isfinite(s)
        cnt_s = np.bincount(rp[ok], minlength=n)
        tot = np.bincount(rp[ok], s[ok], n)
        has = closed & (cnt_s > 0)
        inside[has] = tot[has] / cnt_s[has]
        inside_surface = inside.copy()

        with np.errstate(invalid="ignore", divide="ignore"):
            cen = np.stack([np.bincount(rid, tet * (p0 + p1 + p2)[:, a] / 4.0, n)
                            for a in range(3)], axis=1) / signed[:, None]
        cl = np.flatnonzero(closed & np.isfinite(cen).all(axis=1))
        inside_centroid = np.full(n, np.nan)
        inside_centroid[cl] = sampler(cen[cl])
        if voxel_size:
            tiny = closed & np.isfinite(inside_centroid) \
                & ((6.0 * vol / np.pi) ** (1.0 / 3.0) < 4.0 * voxel_size)
            inside[tiny] = inside_centroid[tiny]
    else:
        inside_surface, inside_centroid = inside.copy(), np.full(n, np.nan)
    return {"vol": vol, "area": area, "closed": closed, "inside": inside,
            "inside_surface": inside_surface, "inside_centroid": inside_centroid,
            "signed": signed, "source": source}


PHASE_METRICS = [  # (per-fluid time-series CSV column, summary variable template)
    ("n_drops", "Number of Drops n_drops ({} drops) [-]"),
    ("d32_m", "Sauter Diameter d32 (VOF, {} drops) [m]"),
    ("d32_dx", "Sauter Diameter d32 / Voxel Size (VOF, {} drops) [-]"),
    ("d10_m", "Mean Drop Diameter d10 (VOF, {} drops) [m]"),
    ("d43_m", "Volume-Mean Drop Diameter d43 (VOF, {} drops) [m]"),
    ("d_max_m", "Largest Drop Diameter (VOF, {} drops) [m]"),
    ("F_disp", "Dispersed Fraction F_disp ({} in drops / {} volume) [-]"),
]


def vof_structures(case_dir: Path, min_voxels: float, voxel_size: float,
                   t_start: float, fluids: list[tuple[int, str, np.ndarray, np.ndarray]],
                   a_int: tuple[np.ndarray, np.ndarray] | None = None,
                   out_dir: Path | None = None,
                   min_d_dx: float = 4.0) -> tuple[list[dict], dict]:
    """Per-fluid drop statistics from the Volume VOF Surface .vtp files.

    Drops are closed structures (bulk phases touching walls/shaft/impeller are
    open and excluded) with an equivalent diameter of at least the stricter of
    min_voxels voxels and min_d_dx voxel sizes (smaller ones are excluded as
    unresolved). Each drop is assigned to
    the fluid inside it, sampled from the .vti volume output at the same time
    (else from the triangle orientation). Per fluid k (fluids = the two
    fluid_volumes() entries): n_drops, d32, d32/dx, d10, d43, d_max and
    F_disp,k = V_drops,k / V_k. With a_int = (t, A_int) from the interface stats
    file, coverage = A_drops / A_int is the share of the interface the drop
    statistics describe. Writes vof_structures.csv and vof_dsd.csv to out_dir.
    Returns (summary rows, plot series).
    """
    pv = import_pyvista("VOF structure analysis")
    if pv is None:
        return [], {}
    # VolumeVOFStructures/vof_structures.*.vtp is a point cloud, not the surface
    frames = find_frames(case_dir, ".vtp", lambda n: "vof" in n and "structure" not in n)
    if not frames:
        warn(f"--dispersion: no VOF surface .vtp files (name containing 'VOF', "
             f"excluding Slice outputs) found under {case_dir}; skipping VOF structures")
        return [], {}
    log(f"VOF structures: {len(frames)} frame(s) under {case_dir}")
    d_min = max((6.0 * min_voxels / np.pi) ** (1.0 / 3.0), min_d_dx) * voxel_size
    v_min = np.pi / 6.0 * d_min ** 3
    log(f"  drop filter: d_eq >= {d_min:.3g} m (stricter of {min_voxels:g} voxels "
        f"and {min_d_dx:g} dx); smaller closed structures are excluded")
    vti_frames = [(tv, p) for tv, p in find_frames(case_dir, ".vti") if tv is not None]
    idx1 = fluids[0][0]
    # label by fluid index so batch columns line up across fluid systems
    phases = [(f"fluid{i}", f"Fluid {i}", ft, fv) for i, _, ft, fv in fluids]
    log("  phases: " + ", ".join(f"Fluid {i} = {name}" for i, name, *_ in fluids))

    ts_rows, dsd_rows, dsd_plot = [], [], []
    sources, id_sources, phase_sources = set(), set(), set()
    n_blocks, n_checked, n_disagree, n_cen, n_cen_dis = 1, 0, 0, 0, 0
    for t_name, paths in frames:
        meshes = [pv.read(p) for p in paths]
        t, src = frame_time(meshes[0], t_name)
        if t is None:
            warn(f"  {paths[0].name}: no time in field data or filename; skipped")
            continue
        sources.add(src)
        n_blocks = max(n_blocks, len(meshes))
        surf = meshes[0] if len(meshes) == 1 else pv.merge(meshes)
        vti = next((p for tv, p in vti_frames
                    if abs(tv - t) <= 1e-6 * max(1.0, abs(t))), None)
        sampler = vti_sampler(pv, vti, idx1) if vti else None
        phase_sources.add(f"fluid-{idx1} volume fraction in .vti" if sampler
                          else "triangle orientation (no matching .vti)")
        s = analyze_vof_surface(pv, surf, sampler, voxel_size)
        id_sources.add(s["source"])
        vol, area, closed = s["vol"], s["area"], s["closed"]

        debris = closed & (vol < v_min)
        kept = closed & ~debris
        is1 = s["inside"] > 0.5
        if sampler:
            n_checked += int(kept.sum())
            n_disagree += int((kept & (is1 != (s["signed"] > 0))).sum())
            both = kept & np.isfinite(s["inside_centroid"])
            n_cen += int(both.sum())
            n_cen_dis += int((both & ((s["inside_centroid"] > 0.5)
                                      != (s["inside_surface"] > 0.5))).sum())
        a_drops = float(area[kept].sum())
        rec = {"time_s": t, "n_regions": len(vol), "n_open": int((~closed).sum()),
               "n_debris": int(debris.sum()), "n_resolved": int(kept.sum()),
               "A_surface_m2": float(area.sum()),
               "A_drops_m2": a_drops, "coverage": np.nan}
        if a_int is not None:
            a_tot = float(np.interp(t, *a_int))
            rec["A_int_m2"] = a_tot
            if a_tot > 0:
                rec["coverage"] = a_drops / a_tot
        for k, (tag, name, ft, fv) in enumerate(phases):
            sel = kept & (is1 if k == 0 else ~is1)
            V, A = vol[sel], area[sel]
            d = (6.0 * V / np.pi) ** (1.0 / 3.0)
            v_phase = float(np.interp(t, ft, fv))
            rec[f"n_drops_{tag}"] = int(sel.sum())
            rec[f"V_drops_m3_{tag}"] = float(V.sum())
            rec[f"F_disp_{tag}"] = V.sum() / v_phase if v_phase > 0 else np.nan
            if V.size:
                d32 = 6.0 * V.sum() / A.sum()
                rec.update({f"d32_m_{tag}": d32, f"d32_dx_{tag}": d32 / voxel_size,
                            f"d10_m_{tag}": float(d.mean()),
                            f"d43_m_{tag}": float((d ** 4).sum() / (d ** 3).sum()),
                            f"d_max_m_{tag}": float(d.max())})
            else:
                rec.update({f"{c}_{tag}": np.nan for c in ("d32_m", "d32_dx", "d10_m",
                                                            "d43_m", "d_max_m")})
            for gid, di, vi, ai in zip(np.flatnonzero(sel), d, V, A):
                dsd_rows.append({"time_s": t, "phase": name, "region_id": int(gid),
                                 "d_eq_m": di, "volume_m3": vi, "area_m2": ai})
                dsd_plot.append((t, di, vi, name))
        ts_rows.append(rec)
    if not ts_rows:
        return [], {}

    ts_rows.sort(key=lambda r: r["time_s"])
    log(f"  frame time taken from: {', '.join(sorted(sources))}")
    log(f"  structures identified by: {', '.join(sorted(id_sources))}")
    log(f"  drop phase from: {', '.join(sorted(phase_sources))}")
    if n_checked:
        msg = (f"  .vti phase disagrees with triangle orientation for "
               f"{n_disagree} of {n_checked} drops")
        (warn if n_disagree > 0.05 * n_checked else log)(msg)
    if n_cen:
        log(f"  centroid and surface-offset phase samples disagree for {n_cen_dis} "
            f"of {n_cen} drops (centroid used below 4 voxels across)")
    log(f"  drops per frame on average: "
        f"{np.mean([r['n_resolved'] for r in ts_rows]):.1f} resolved, "
        f"{np.mean([r['n_debris'] for r in ts_rows]):.1f} excluded (d_eq < {d_min:.3g} m)")
    n_after = sum(r["time_s"] >= t_start for r in ts_rows)
    if n_after < 10:
        warn(f"  only {n_after} VOF frame(s) at t >= {t_start:g} s; drop statistics "
             "are poorly averaged (write more Volume VOF Surface output)")
    mean_open = np.mean([r["n_open"] for r in ts_rows])
    log(f"  open structures (bulk phases, wall-attached drops/ligaments; excluded): "
        f"{mean_open:.1f} per frame on average")
    if n_blocks > 1:
        warn(f"  VOF surface written in up to {n_blocks} blocks per frame; drops "
             "crossing block seams whose vertices do not coincide are open and "
             "excluded, biasing d32 toward small drops")

    out_dir = out_dir or case_dir
    cols = ["time_s", "n_regions", "n_open", "n_debris", "n_resolved",
            "A_surface_m2", "A_drops_m2"] \
        + (["A_int_m2"] if a_int is not None else []) + ["coverage"] \
        + [f"{c}_{tag}" for tag, *_ in phases
           for c in ("n_drops", "V_drops_m3", "F_disp", "d32_m", "d32_dx",
                     "d10_m", "d43_m", "d_max_m")]
    write_csv(ts_rows, out_dir / "vof_structures.csv", columns=cols, what="frames")
    write_csv(dsd_rows, out_dir / "vof_dsd.csv",
              columns=["time_s", "phase", "region_id", "d_eq_m", "volume_m3", "area_m2"],
              what="structures")

    t = np.array([r["time_s"] for r in ts_rows])
    series = {"t": t, "dsd": dsd_plot, "phases": [(tag, name) for tag, name, *_ in phases]}
    for c in cols[1:]:
        series[c] = np.array([r.get(c, np.nan) for r in ts_rows], float)
    rows = []
    metrics = [(f"{key}_{tag}", var.format(name, name))
               for tag, name, *_ in phases for key, var in PHASE_METRICS]
    if a_int is not None:
        metrics.append(("coverage", "Drop Interface Coverage A_drops/A_int [-]"))
    for key, var in metrics:
        row = series_row("vof_structures.csv", var, t, series[key], t_start, "vof")
        if row:
            rows.append(row)
            log(f"  {var}: {row['mean']:.5g} \u00b1 {row['std']:.3g}")
            if key == "coverage" and row["mean"] < 0.1:
                warn(f"  closed drops carry only {100 * row['mean']:.2g}% of the "
                     "interfacial area (rest: bulk interface, ligaments, "
                     "wall-attached structures); drop statistics describe a small "
                     "part of the dispersion, compare cases by the specific "
                     "interfacial area a")
    return rows, series


def region_shares(ndimage, blocks: list[tuple]) -> list[float] | None:
    """Per fluid: voxels in the largest connected region of phi_k > 0.5 / all
    phi_k > 0.5 voxels, with the blocks merged onto one grid (None if the
    blocks have different spacings)."""
    spacing = blocks[0][5]
    if any(not np.allclose(b[5], spacing) for b in blocks):
        return None
    org = np.min([b[4] for b in blocks], axis=0)
    offs = [np.rint((b[4] - org) / spacing).astype(int) for b in blocks]
    gdims = np.max([o + b[6] for o, b in zip(offs, blocks)], axis=0)
    shares = []
    for k in range(blocks[0][0].shape[0]):
        grid = np.zeros(tuple(gdims[::-1]), bool)
        for o, (phi, _, fluid, _, _, _, dims) in zip(offs, blocks):
            grid[o[2]:o[2] + dims[2], o[1]:o[1] + dims[1], o[0]:o[0] + dims[0]] \
                |= (phi[k] > 0.5) & fluid
        lab, _ = ndimage.label(grid)
        sizes = np.bincount(lab.ravel())[1:]
        shares.append(float(sizes.max() / sizes.sum()) if sizes.size and sizes.sum()
                      else np.nan)
    return shares


def phase_uniformity(case_dir: Path, n_zones_z: int, n_zones_r: int, t_start: float,
                     vertical_axis: str = "y", axis_center: list[float] | None = None,
                     out_dir: Path | None = None,
                     zone_scales: list[tuple[int, int]] | None = None,
                     window: float = 1.0, tol: float = 0.02,
                     continuity_threshold: float = 0.9) -> tuple[list[dict], dict]:
    """Zonal CoV and segregation index of the phase fraction, vertical holdup
    profile and phase continuity from .vti output.

    phi is coarse-grained into NZ axial x NR radial zones per zone scale
    (default [(n_zones_z, n_zones_r)]) before computing statistics (voxel phi
    is ~0/1). With two 'Fluid N ... Volume Fraction' arrays, also per fluid:
    segregation index S = std/sqrt(mean*(1-mean)) (0 homogeneous, 1 fully
    segregated), homogeneity time t_S (S within 5% of its plateau) and, with
    scipy, the largest connected region share of phi_k > 0.5, giving the
    continuous phase (share > continuity_threshold). Writes
    phase_uniformity.csv, holdup_profile.csv and phase_continuity.csv.
    """
    pv = import_pyvista("phase uniformity analysis")
    if pv is None:
        return [], {}
    frames = find_frames(case_dir, ".vti", lambda n: not n.startswith("boundaryconditions"))
    if not frames:
        warn(f"--uniformity: no .vti volume files found under {case_dir}; skipping")
        return [], {}
    scales = zone_scales or [(n_zones_z, n_zones_r)]
    ax = "xyz".index(vertical_axis)
    h1, h2 = [a for a in range(3) if a != ax]
    log(f"phase uniformity: {len(frames)} frame(s) under {case_dir}, "
        f"vertical axis {vertical_axis}, zone scales "
        + ", ".join(f"{a}x{b}" for a, b in scales))

    phi_name = mask_name = ndimage = None
    frac_names: list[str] = []
    phases: list[tuple[str, str, str]] = []  # (array, column suffix, label)
    fnames: dict[str, str] = {}
    seen: dict[str, list[str]] = {}
    cov_rows, prof_rows, cont_rows, sources = [], [], [], set()
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
                         if any(k in n.lower() for k in ("vof", "phase", "fraction"))
                         and "structure" not in n.lower()]
                if not cands:
                    seen.setdefault(p.parent.parent.name, names)
                    continue
                masks = [n for n in names if any(k in n.lower() for k in ("solid", "flag"))]
                mask_name = masks[0] if masks else None
                frac_names = [n for n in cell_names
                              if re.search(r"fluid\s*\d+.*volume fraction", n, re.I)]
                if len(frac_names) >= 2:
                    for n in frac_names:
                        m = re.search(r"fluid\s*(\d+)\s*(?:\(([^)]*)\))?", n, re.I)
                        phases.append((n, f"_fluid{m.group(1)}", f"Fluid {m.group(1)}"))
                        fnames[f"Fluid {m.group(1)}"] = (m.group(2) or "").strip()
                    try:
                        from scipy import ndimage
                    except ImportError:
                        warn("--uniformity: scipy is not installed (pip install scipy); "
                             "skipping phase continuity")
                else:
                    if len(cands) > 1:
                        warn(f"--uniformity: ambiguous phase-fraction arrays "
                             f"{cands}; using '{cands[0]}'")
                    phases.append((cands[0], "", ""))
                    log("  single phase-fraction array (free-surface or one fluid): "
                        "segregation index and phase continuity skipped")
                    if mask_name is None:
                        frac_names = []
                phi_name = phases[0][0]
                log(f"  phase fraction: " + ", ".join(f"'{n}'" for n, *_ in phases)
                    + f" from {p.parent.parent.name}"
                    + (f", non-fluid mask: '{mask_name}' (nonzero = solid)" if mask_name
                       else ", non-fluid mask: sum of fluid volume fractions < 0.5"
                       if frac_names else ""))
            if phi_name not in names:
                continue
            reader.disable_all_cell_arrays()
            reader.disable_all_point_arrays()
            for n in {mask_name, *frac_names, *(n for n, *_ in phases)} - {None}:
                (reader.enable_cell_array if n in cell_names else reader.enable_point_array)(n)
            mesh = reader.read()
            t, src = frame_time(mesh, t_name)
            bounds.append(mesh.bounds)

            on_cells = phi_name in cell_names
            dims = np.array(mesh.dimensions) - (1 if on_cells else 0)
            off = 0.5 if on_cells else 0.0
            data = mesh.cell_data if on_cells else mesh.point_data
            shape = tuple(dims[::-1])  # VTK order: x fastest
            phi = np.stack([np.asarray(data[n], float).reshape(shape) for n, *_ in phases])
            fluid = np.isfinite(phi).all(axis=0)
            if mask_name is not None and mask_name in data.keys():
                fluid &= np.asarray(data[mask_name]).reshape(shape) == 0
            elif frac_names and all(n in data.keys() for n in frac_names):
                # M-Star writes no solid flag; solid cells have all fluid fractions = 0
                total = sum(np.asarray(data[n], float) for n in frac_names)
                fluid &= total.reshape(shape) > 0.5
            c1d = [mesh.origin[a] + (np.arange(dims[a]) + off) * mesh.spacing[a]
                   for a in range(3)]
            zz, yy, xx = np.meshgrid(c1d[2], c1d[1], c1d[0], indexing="ij", sparse=True)
            coord = (xx, yy, zz)
            blocks.append((phi, coord, fluid, mesh.spacing[ax],
                           np.array(mesh.origin) + off * np.array(mesh.spacing),
                           np.array(mesh.spacing), dims))
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
        for phi, coord, fluid, *_ in blocks:
            ph.append(phi[:, fluid])
            vz.append(np.broadcast_to(coord[ax], fluid.shape)[fluid])
            rr.append(np.broadcast_to(np.sqrt((coord[h1] - c1) ** 2 + (coord[h2] - c2) ** 2),
                                      fluid.shape)[fluid])
        ph, vz, rr = np.concatenate(ph, axis=1), np.concatenate(vz), np.concatenate(rr)
        if vz.size == 0:
            warn(f"  t = {t:g} s: no fluid voxels; skipped")
            continue

        dz = blocks[0][3]
        z0, z1, r1 = vz.min() - dz / 2, vz.max() + dz / 2, rr.max()
        two_fluid = bool(phases[0][2])
        rec = {"time_s": t}
        for (_, sfx, _), p in zip(phases, ph):
            rec[f"phi_mean{sfx}"] = float(p.mean())
        for nzz, nrr in scales:
            ssfx = f"_{nzz}x{nrr}"
            iz = np.minimum(((vz - z0) / (z1 - z0) * nzz).astype(int), nzz - 1)
            ir = np.minimum((rr / max(r1, 1e-30) * nrr).astype(int), nrr - 1)
            zone = iz * nrr + ir
            w = np.bincount(zone, minlength=nzz * nrr).astype(float)
            occ = w > 0
            rec[f"n_zones{ssfx}"] = int(occ.sum())
            for (_, sfx, _), p in zip(phases, ph):
                phi_zone = np.bincount(zone, p, minlength=nzz * nrr)[occ] / w[occ]
                mean = np.average(phi_zone, weights=w[occ])
                std = np.sqrt(np.average((phi_zone - mean) ** 2, weights=w[occ]))
                rec[f"cov{sfx}{ssfx}"] = std / mean if mean else np.nan
                if two_fluid:
                    rec[f"S{sfx}{ssfx}"] = (std / np.sqrt(mean * (1 - mean))
                                            if 0 < mean < 1 else np.nan)
        cov_rows.append(rec)

        layer = np.rint((vz - vz.min()) / dz).astype(int)
        lw = np.bincount(layer)
        prof = {k: {"time_s": t, "z_m": vz.min() + k * dz} for k in np.flatnonzero(lw)}
        for (_, sfx, _), p in zip(phases, ph):
            lsum = np.bincount(layer, p)
            for k, r in prof.items():
                r[f"phi_mean{sfx}"] = lsum[k] / lw[k]
        prof_rows.extend(prof.values())

        if two_fluid and ndimage is not None:
            shares = region_shares(ndimage, blocks)
            if shares is None:
                warn("  .vti blocks have different spacings; phase continuity skipped")
                ndimage = None
            else:
                cont = [lab for (_, _, lab), s in zip(phases, shares)
                        if s > continuity_threshold]
                crec = {"time_s": t, "continuous_phase":
                        "both (stratified or co-continuous)" if len(cont) > 1
                        else f"{cont[0]} ({fnames[cont[0]]})" if cont else "none"}
                crec.update({f"share{sfx}": s for (_, sfx, _), s in zip(phases, shares)})
                cont_rows.append(crec)
    if phi_name is None:
        warn("--uniformity: no phase-fraction array (name containing 'vof', 'phase' "
             "or 'fraction') in any .vti output; enable the volume fraction in the "
             "M-Star Volume output. Found: "
             + "; ".join(f"{d}: {', '.join(n)}" for d, n in seen.items()))
        return [], {}
    if not cov_rows:
        return [], {}

    cov_rows.sort(key=lambda r: r["time_s"])
    cont_rows.sort(key=lambda r: r["time_s"])
    log(f"  frame time taken from: {', '.join(sorted(sources))}")
    out_dir = out_dir or case_dir
    two_fluid = bool(phases[0][2])
    sfxs = [sfx for _, sfx, _ in phases]
    combos = [(f"{sfx}_{a}x{b}", ", ".join(x for x in (label, f"{a}x{b}") if x))
              for a, b in scales for _, sfx, label in phases]
    write_csv(cov_rows, out_dir / "phase_uniformity.csv",
              columns=["time_s"] + [f"phi_mean{s}" for s in sfxs]
              + [f"n_zones_{a}x{b}" for a, b in scales]
              + [f"cov{k}" for k, _ in combos]
              + ([f"S{k}" for k, _ in combos] if two_fluid else []), what="frames")
    write_csv(prof_rows, out_dir / "holdup_profile.csv",
              columns=["time_s", "z_m"] + [f"phi_mean{s}" for s in sfxs], what="rows")

    t = np.array([r["time_s"] for r in cov_rows])
    n_after = int((t >= t_start).sum())
    if n_after < 10:
        warn(f"  only {n_after} .vti frame(s) at t >= {t_start:g} s; uniformity "
             "statistics are poorly averaged (write more Volume output)")
    # detect_steady_start needs >= 3 samples per window; VTK frames are sparse
    window_f = max(window, 3 * float(np.median(np.diff(t)))) if t.size > 1 else window
    series = {"t": t, "profile": prof_rows, "combos": combos,
              "phases": [(sfx, label) for _, sfx, label in phases]}
    rows = []
    for key, label in combos:
        series[f"cov{key}"] = np.array([r[f"cov{key}"] for r in cov_rows], float)
        row = series_row("phase_uniformity.csv",
                         f"Zonal CoV of Phase Fraction ({label}) [-]",
                         t, series[f"cov{key}"], t_start, "zonal")
        if row:
            rows.append(row)
        if not two_fluid:
            continue
        s = np.array([r[f"S{key}"] for r in cov_rows], float)
        series[f"S{key}"] = s
        row = series_row("phase_uniformity.csv", f"Segregation Index S ({label}) [-]",
                         t, s, t_start, "zonal")
        if row:
            rows.append(row)
        kin = approach_kinetics(t, s, window_f, tol, levels=(0.95,))
        if kin and np.isfinite(kin["t95"]):
            rows.append(value_row("phase_uniformity.csv",
                                  f"Homogeneity Time t_S ({label}) [s]", kin["t95"],
                                  "S(t) kinetics", kin["t_ss"], float(t[-1]), int(t.size)))
        elif kin:
            warn(f"  S(t) ({label}) never plateaus; no homogeneity time")

    if cont_rows:
        ct = np.array([r["time_s"] for r in cont_rows])
        write_csv(cont_rows, out_dir / "phase_continuity.csv",
                  columns=["time_s"] + [f"share{s}" for s in sfxs] + ["continuous_phase"],
                  what="frames")
        series["share"] = {}
        for _, sfx, label in phases:
            series["share"][label] = np.array([r[f"share{sfx}"] for r in cont_rows])
            row = series_row("phase_continuity.csv", f"Largest Region Share ({label}) [-]",
                             ct, series["share"][label], t_start, "zonal")
            if row:
                rows.append(row)
        series["share_t"] = ct
        labels = [r["continuous_phase"] for r in cont_rows if r["time_s"] >= t_start] \
            or [r["continuous_phase"] for r in cont_rows]
        from collections import Counter
        main_label = Counter(labels).most_common(1)[0][0]
        rows.append(value_row("phase_continuity.csv", "Continuous Phase", main_label,
                              "zonal", t_start, float(ct[-1]), len(labels)))
        if len(set(labels)) > 1:
            warn(f"  continuous phase changes at t >= {t_start:g} s "
                 f"({', '.join(sorted(set(labels)))}): possible phase inversion")
        log(f"  continuous phase: {main_label}")
    for r in rows:
        if isinstance(r["mean"], float):
            log(f"  {r['variable']}: {r['mean']:.4g}"
                + (f" \u00b1 {r['std']:.3g}" if r["std"] != "" else ""))
    return rows, series


def resolution_check(files: list[Path], t_start: float, voxel_size: float,
                     sigma: float | None, rho_c: float | None,
                     c_hinze: float = 0.725, k_phi: float = 0.0, eps_ratio: float = 1.0,
                     eps_file: str | None = None, phi: float | None = None) -> list[dict]:
    """Hinze maximum stable drop size vs voxel size,
    d_max = C*(1 + k_phi*phi)*(sigma/rho_c)^0.6*(eps_ratio*eps)^-0.4, with eps the
    steady mean energy dissipation rate from Fluid.txt or eps_file."""
    if sigma is None or rho_c is None:
        warn("--dispersion: resolution check needs --sigma and --rho-continuous; skipped")
        return []
    if eps_file:
        p = Path(eps_file)
        src = [p] if p.is_file() else [f for f in files if eps_file.lower() in str(f).lower()]
        if not src:
            warn(f"--eps-file '{eps_file}' matched no stats file; resolution check skipped")
            return []
    else:
        src = [f for f in files if f.name.lower() == "fluid.txt"]
    table = load_stats_table(src[0]) if src else None
    if table is None:
        warn(f"--dispersion: no {'--eps-file' if eps_file else 'Fluid.txt'} stats "
             "table for the resolution check; skipped")
        return []
    header, data = table
    src_name = f"{src[0].parent.name}/{src[0].name}" if eps_file else src[0].name
    cands = [j for j in range(1, len(header)) if "energy dissipation" in norm_name(header[j])]
    if not cands:
        warn(f"--dispersion: no energy dissipation column in {src_name}; "
             "resolution check skipped")
        return []
    # prefer the instantaneous per-unit-mass value
    j = min(cands, key=lambda j: (var_unit(header[j]).lower() != "w/kg",
                                  "time-avg" in header[j].lower()))
    row = series_row(src_name, header[j].strip(), data[:, 0], data[:, j],
                     t_start, "hinze")
    if row is None or row["mean"] <= 0:
        warn("--dispersion: no positive steady energy dissipation; resolution check skipped")
        return []
    eps = row["mean"]
    if k_phi and phi is None:
        warn("--k-phi given but the dispersed fraction is unknown; holdup correction skipped")
    phi_eff = phi if phi is not None else 0.0
    d_hinze = c_hinze * (1 + k_phi * phi_eff) * (sigma / rho_c) ** 0.6 \
        * (eps_ratio * eps) ** -0.4
    ratio = d_hinze / voxel_size
    log(f"resolution check: eps = {eps:.4g} from '{header[j].strip()}' in {src_name}, "
        f"eps_ratio = {eps_ratio:g}, holdup factor 1 + {k_phi:g}*{phi_eff:.3g}; "
        "resolved eps excludes subgrid dissipation, so d_max,Hinze is an upper estimate")
    log(f"  d_max,Hinze = {d_hinze:.4g} m = {ratio:.3g} voxels")
    if ratio < 5:
        warn(f"Hinze maximum stable drop size is only {ratio:.3g} voxels (< 5): "
             "drop sizes are under-resolved, DSD/d32 results reflect the grid "
             "scale; the specific interfacial area and holdup remain meaningful")
    common = {"file": src_name, "steady_method": "hinze",
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
    kin = itf.get("kinetics", {})
    if kin and np.isfinite(kin.get("a_ss", np.nan)) and kin["a_ss"] != kin["a0"]:
        marks = [(kin["t95"], f"t95 = {kin['t95']:.3g} s")] \
            if np.isfinite(kin.get("t95", np.nan)) else []
        panels.append(("Normalized a(t): (a - a0)/(a_ss - a0)",
                       [line(itf["t"], (itf["a"] - kin["a0"]) / (kin["a_ss"] - kin["a0"]),
                             "normalized a")], True, "-", marks))
    d32 = []
    if "d32" in itf:
        d32.append(line(itf["t"], itf["d32"], "d32 (interface)"))
    phases = vof.get("phases", [])
    for tag, label in phases:
        d32.append(line(vof["t"], vof[f"d32_m_{tag}"], f"d32 ({label} drops)"))
    if d32:
        panels.append(("Sauter diameter d32(t)", d32, True, "m"))
    if phases:
        panels.append(("Dispersed fraction F_disp(t) = V_drops / V_fluid",
                       [line(vof["t"], vof[f"F_disp_{tag}"], label)
                        for tag, label in phases], True, "-"))
        panels.append(("Number of drops n_drops(t)",
                       [line(vof["t"], vof[f"n_drops_{tag}"], label)
                        for tag, label in phases], True, "-"))
    if np.isfinite(vof.get("coverage", [np.nan])).any():
        panels.append(("Drop interface coverage A_drops/A_int (t)",
                       [line(vof["t"], vof["coverage"], "coverage")], True, "-"))
    if uni.get("combos"):
        panels.append(("Zonal CoV of phase fraction (t)",
                       [line(uni["t"], uni[f"cov{key}"], f"CoV {label}")
                        for key, label in uni["combos"]], True, "-"))
        s_lines = [line(uni["t"], uni[f"S{key}"], f"S {label}")
                   for key, label in uni["combos"] if f"S{key}" in uni]
        if s_lines:
            panels.append(("Segregation index S(t) (0 homogeneous, 1 segregated)",
                           s_lines, True, "-"))
    if uni.get("share"):
        panels.append(("Largest connected region share (t)",
                       [line(uni["share_t"], v, label) for label, v in uni["share"].items()],
                       True, "-"))

    dsd = [(d, label) for t, d, v, label in vof.get("dsd", []) if t >= t_start]
    if dsd:
        d = np.array([x[0] for x in dsd])
        edges = np.histogram_bin_edges(d, bins=min(40, max(5, int(np.sqrt(d.size)))))
        centers, width = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
        bars = []
        for _, label in phases:
            num, _ = np.histogram([x[0] for x in dsd if x[1] == label], edges)
            if num.sum():
                bars.append(go.Bar(x=centers, y=num / num.sum(), width=width,
                                   name=f"{label} drops", opacity=0.6))
        panels.append(("Steady-state DSD (number-weighted, per fluid)", bars,
                       False, "fraction"))

    if uni.get("profile"):
        prof = uni["profile"]
        # with two fluids phi_2 = 1 - phi_1, so one heatmap suffices
        sfx, label = uni["phases"][-1]
        times = sorted({r["time_s"] for r in prof})
        zs = np.unique(np.round([r["z_m"] for r in prof], 9))
        grid = np.full((zs.size, len(times)), np.nan)
        ti = {t: i for i, t in enumerate(times)}
        for r in prof:
            grid[np.searchsorted(zs, round(r["z_m"], 9)), ti[r["time_s"]]] = r[f"phi_mean{sfx}"]
        panels.append((f"Holdup profile \u03c6(z, t) {label}".strip(), [
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
    for i, (title, traces, timeseries, ytitle, *marks) in enumerate(panels):
        r, c = divmod(i, ncols)
        r, c = r + 1, c + 1
        for tr in traces:
            fig.add_trace(tr, row=r, col=c)
        if timeseries:
            fig.add_vline(x=t_start, line_dash="dash", line_color="green",
                          line_width=1, row=r, col=c)
            for x, text in (marks[0] if marks else []):
                fig.add_vline(x=x, line_dash="dot", line_color="purple", line_width=1,
                              annotation_text=text, annotation_font_size=9, row=r, col=c)
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

def process_case(case: Path, stats_dir: Path, args, output: Path | None = None,
                 plot_dir: Path | None = None) -> tuple[list[dict], dict[str, list[dict]]]:
    """Summarize one case; write its CSVs (and plots). Returns (all rows,
    {"dispersion": rows, "uniformity": rows}); ([], {}) on failure."""
    files = find_stats_files(stats_dir)
    if not files:
        warn(f"no .txt stats files found in {stats_dir}")
        return [], {}
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
            return [], {}
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
        return [], {}

    rows_mix, mix_vlines = scalar_mixing(files, stats_dir, args.mix_rsd)
    rows.extend(rows_mix)

    base = case if case.is_dir() else case.parent
    disp: dict = {}
    disp_rows: list[dict] = []  # evaluated from t_disp
    groups: dict[str, list[dict]] = {"dispersion": [], "uniformity": []}
    t_disp, disp_method = t_start, None
    fluids = fluid_volumes(files)
    voxel = None
    if args.dispersion:
        voxel = args.voxel_size or input_voxel_size(base)
        if voxel is None:
            fail(f"{case}: --dispersion needs --voxel-size (no <dx> found in input.xml)")
        if not args.voxel_size:
            log(f"voxel size: {voxel:g} m (<dx> from input.xml)")
        rows_itf, disp["interface"] = interface_metrics(
            files, stats_dir, t_start, args.dispersed_fraction, args.liquid_volume,
            args.window, args.tol, args.vessel_diameter, args.area_column)
        rows.extend(rows_itf)
        groups["dispersion"].extend(rows_itf)
        t_ss_a = disp["interface"].get("kinetics", {}).get("t_ss_a", np.nan)
        if np.isfinite(t_ss_a):
            t_disp, disp_method = max(t_start, t_ss_a), "a(t) plateau"
    if args.disp_time is not None:
        t_disp, disp_method = args.disp_time, "user"
    if args.dispersion or args.uniformity:
        log(f"steady-state start: flow t >= {t_start:.3f} s, dispersion metrics "
            f"t >= {t_disp:.3f} s ({disp_method or 'flow steady state'})")

    if args.dispersion:
        coverage = None
        if len(fluids) != 2:
            warn("--dispersion: Fluid.txt has no 'Fluid N (name) Volume' columns "
                 "for two fluids (free-surface or single-fluid case); skipping "
                 "drop statistics")
        else:
            vof_root = base
            if args.vof_dir is not None:
                vof_root = args.vof_dir if args.vof_dir.is_absolute() or not args.batch \
                    else base / args.vof_dir
            itf = disp["interface"]
            a_int = (itf["t"], itf["area"]) if "area" in itf else None
            rows_vof, disp["vof"] = vof_structures(vof_root, args.min_voxels, voxel,
                                                   t_disp, fluids, a_int, out_dir=base,
                                                   min_d_dx=args.min_d_dx)
            disp_rows.extend(rows_vof)
            groups["dispersion"].extend(rows_vof)
            coverage = next((r["mean"] for r in rows_vof
                             if r["variable"].startswith("Drop Interface Coverage")), None)
        d32_var = "Sauter Diameter d32 (interface) [m]"
        if any(r["variable"] == d32_var for r in rows):
            if coverage is not None and coverage < 0.5:
                warn(f"--dispersion: area-based d32 = 6*phi/a skipped: closed drops carry "
                     f"only {100 * coverage:.2g}% of the interface (< 50%), so most of "
                     "the interface is not in drops and 6*phi/a is not a drop size")
                rows = [r for r in rows if r["variable"] != d32_var]
                disp["interface"].pop("d32", None)
            elif coverage is None:
                warn("--dispersion: drop interface coverage unknown; area-based d32 "
                     "assumes the whole interface belongs to drops")
        phi_d = args.dispersed_fraction
        if phi_d is None and len(fluids) == 2:
            (_, _, t1, v1), (_, _, t2, v2) = fluids
            frac1 = v1 / (v1 + np.interp(t1, t2, v2))
            sel = t1 >= t_disp if (t1 >= t_disp).any() else slice(None)
            phi_d = float(min(frac1[sel].mean(), 1 - frac1[sel].mean()))
        rows_res = resolution_check(
            files, t_disp, voxel, args.sigma, args.rho_continuous, args.hinze_c,
            args.k_phi, args.eps_ratio, args.eps_file, phi_d)
        disp_rows.extend(rows_res)
        groups["dispersion"].extend(rows_res)
    if args.uniformity:
        scales = args.zone_scales or [(args.zones_z, args.zones_r)]
        rows_uni, disp["uniformity"] = phase_uniformity(
            base, args.zones_z, args.zones_r, t_disp, args.vertical_axis,
            args.axis_center, out_dir=base, zone_scales=scales, window=args.window,
            tol=args.tol, continuity_threshold=args.continuous_threshold)
        disp_rows.extend(rows_uni)
        groups["uniformity"].extend(rows_uni)
    if disp_method:
        for r in disp_rows:
            r["steady_method"], r["steady_start_s"] = disp_method, t_disp
    rows.extend(disp_rows)

    out = output or base / "stats_summary.csv"
    write_csv(rows, out)
    kept = {id(r) for r in rows}  # the area-based d32 may have been dropped
    groups = {k: [r for r in g if id(r) in kept] for k, g in groups.items()}
    for name, enabled in (("dispersion", args.dispersion), ("uniformity", args.uniformity)):
        if not enabled:
            continue
        if groups[name]:
            write_csv(groups[name], out.with_name(f"{out.stem}_{name}{out.suffix}"))
        else:
            warn(f"--{name}: no metrics computed; no {name} summary written")

    if args.plot is not None:
        make_plots(files, stats_dir, args.plot, t_start,
                   plot_dir or base / "stats_plots", mix_vlines=mix_vlines)
        if disp:
            plot_dispersion(disp, t_disp, (plot_dir or base / "stats_plots") / "dispersion.html")
    return rows, groups


def find_batch_cases(parent: Path) -> list[tuple[Path, Path]]:
    """Return (case dir, stats dir) for each immediate, non-hidden subfolder
    with stats files."""
    cases = []
    for sub in sorted(p for p in parent.iterdir()
                      if p.is_dir() and not p.name.startswith(".")):
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
                         "dispersed fraction, n_drops, cov, coverage, hinze, "
                         "interface enhancement, dispersion time, "
                         "time-integrated, segregation, homogeneity time, "
                         "continuous phase)")
    ap.add_argument("--ref-case", default=None, metavar="CASE",
                    help="reference case (exact or unique substring) for the %% "
                         "differences in the means bar plots (default: first case)")
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
                    help="interfacial-area metrics and a(t) dispersion kinetics "
                         "(E = a_ss/a0, t90/t95, tau, plateau CoV, integral of a) "
                         "from the interface stats file, "
                         "per-fluid drop statistics (n_drops, d32, d32/dx, "
                         "F_disp = V_drops/V_fluid) and interface coverage "
                         "A_drops/A_int from the Volume VOF Surface .vtp output "
                         "(Immiscible Two-Fluid only), and (with --sigma, "
                         "--rho-continuous) a Hinze resolution check. VTK-based "
                         "metrics start at the a(t) plateau (see --disp-time)")
    dg.add_argument("--disp-time", type=float, default=None, metavar="T",
                    help="start time [s] for the dispersion/uniformity metrics "
                         "(default: max(flow steady state, a(t) plateau start))")
    dg.add_argument("--area-column", default=None, metavar="NAME",
                    help="interfacial area column (fuzzy) in the interface stats "
                         "file (default: the first column containing 'area')")
    dg.add_argument("--vessel-diameter", type=float, default=None, metavar="D",
                    help="vessel diameter [m]; a0 is checked against the flat-"
                         "interface estimate pi*D^2/4/V_liq")
    dg.add_argument("--dispersed-fraction", type=float, default=None, metavar="PHI",
                    help="dispersed-phase volume fraction, for the area-based "
                         "d32 = 6*PHI/a (reported only if closed drops carry "
                         ">= 50%% of the interface) and the Hinze holdup "
                         "correction (default there: minority fraction from Fluid.txt)")
    dg.add_argument("--liquid-volume", type=float, default=None, metavar="M3",
                    help="liquid volume [m^3] for a = A_int/V_liq "
                         "(default: 'Fluid Volume' from Fluid.txt)")
    dg.add_argument("--vof-dir", type=Path, default=None,
                    help="directory searched recursively for VOF surface .vtp "
                         "files (default: the case directory; relative to each "
                         "case with --batch)")
    dg.add_argument("--voxel-size", type=float, default=None, metavar="M",
                    help="lattice spacing [m] for the debris filter, d32/dx and "
                         "the resolution check; required with --dispersion "
                         "unless <dx> is found in the case's input.xml")
    dg.add_argument("--min-voxels", type=float, default=8,
                    help="VOF structures smaller than this many voxels are "
                         "dropped as numerical debris")
    dg.add_argument("--min-d-dx", type=float, default=4,
                    help="drops with equivalent diameter < MIN_D_DX voxel sizes "
                         "are excluded as unresolved (the stricter of this and "
                         "--min-voxels applies)")
    dg.add_argument("--sigma", type=float, default=None,
                    help="interfacial tension [N/m] for the Hinze resolution check")
    dg.add_argument("--rho-continuous", type=float, default=None, metavar="RHO",
                    help="continuous-phase density [kg/m^3] for the Hinze "
                         "resolution check")
    dg.add_argument("--hinze-c", type=float, default=0.725,
                    help="Hinze constant C in d_max = C*(1 + K_PHI*phi)*"
                         "(sigma/rho_c)^0.6*(EPS_RATIO*eps)^-0.4")
    dg.add_argument("--k-phi", type=float, default=0.0,
                    help="holdup (coalescence) correction factor in the Hinze check")
    dg.add_argument("--eps-ratio", type=float, default=1.0,
                    help="local/mean energy dissipation ratio in the Hinze check, "
                         "e.g. impeller-region eps / mean eps")
    dg.add_argument("--eps-file", default=None, metavar="FILE",
                    help="stats file (path or name fragment, e.g. an impeller "
                         "ControlVolume) to take eps from for the Hinze check "
                         "(default: Fluid.txt)")
    dg.add_argument("--uniformity", action="store_true",
                    help="zonal CoV and vertical holdup profile of the phase "
                         "fraction from the 3D Volume .vti output; for two "
                         "fluids also the segregation index S, homogeneity time "
                         "t_S and the continuous phase (needs scipy)")
    dg.add_argument("--zone-scales", nargs="+", default=None, metavar="NZxNR",
                    help="zone scales for CoV and S, e.g. 5x2 10x3 20x5 "
                         "(default: ZONES_Z x ZONES_R = 10x3)")
    dg.add_argument("--continuous-threshold", type=float, default=0.9,
                    help="a fluid is continuous if its largest connected region "
                         "holds more than this share of its volume")
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
    if args.zone_scales:
        scales = []
        for s in args.zone_scales:
            m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", s)
            if not m or int(m.group(1)) < 1 or int(m.group(2)) < 1:
                fail(f"--zone-scales: '{s}' is not NZxNR with NZ, NR >= 1")
            scales.append((int(m.group(1)), int(m.group(2))))
        args.zone_scales = scales
    if not 0 < args.continuous_threshold <= 1:
        fail("--continuous-threshold must be in (0, 1]")

    if args.batch:
        if not args.case.is_dir():
            fail(f"--batch requires a directory: {args.case}")
        cases = find_batch_cases(args.case)
        if not cases:
            fail(f"no case subfolders with stats found under {args.case}")
        log(f"batch mode: {len(cases)} case(s) under {args.case}")

        batch_rows = []
        batch_groups: dict[str, list[dict]] = {}
        for case, stats_dir in cases:
            log(f"=== case: {case.name} ===")
            rows, groups = process_case(case, stats_dir, args)
            for r in rows:
                batch_rows.append({"case": case.name, **r})
            for name, g in groups.items():
                batch_groups.setdefault(name, []).extend({"case": case.name, **r} for r in g)
        if not batch_rows:
            fail("no variables summarized in any case")
        out = args.output or args.case / "batch_summary.csv"
        write_csv(batch_rows, out, columns=["case"] + COLUMNS)
        write_wide_csv(batch_rows, out.with_name(f"{out.stem}_means{out.suffix}"),
                       args.means_vars or DEFAULT_MEANS_VARS,
                       warn_unmatched=args.means_vars is not None,
                       exclude=None if args.means_vars else DEFAULT_MEANS_EXCLUDE,
                       ref_case=args.ref_case)
        for name, g in batch_groups.items():
            if g:
                write_csv(g, out.with_name(f"{out.stem}_{name}{out.suffix}"),
                          columns=["case"] + COLUMNS)
                write_wide_csv(g, out.with_name(f"{out.stem}_{name}_means{out.suffix}"),
                               [""], warn_unmatched=False, ref_case=args.ref_case)
        write_settings_csv([case_settings(c, sd) for c, sd in cases],
                           out.with_name(f"{out.stem}_settings{out.suffix}"))
    else:
        stats_dir = find_stats_dir(args.case)
        base = args.case if args.case.is_dir() else args.case.parent
        out = args.output or base / "stats_summary.csv"
        rows, groups = process_case(args.case, stats_dir, args,
                                    output=out, plot_dir=args.plot_dir)
        if not rows:
            fail("no variables summarized")
        write_wide_csv([{"case": args.case.name, **r} for r in rows],
                       out.with_name(f"{out.stem}_means{out.suffix}"),
                       args.means_vars or DEFAULT_MEANS_VARS,
                       warn_unmatched=args.means_vars is not None,
                       exclude=None if args.means_vars else DEFAULT_MEANS_EXCLUDE)
        for name, g in groups.items():
            if g:
                write_wide_csv([{"case": args.case.name, **r} for r in g],
                               out.with_name(f"{out.stem}_{name}_means{out.suffix}"),
                               [""], warn_unmatched=False)
        write_settings_csv([case_settings(args.case, stats_dir)],
                           out.with_name(f"{out.stem}_settings{out.suffix}"))
    log("done")


if __name__ == "__main__":
    main()
