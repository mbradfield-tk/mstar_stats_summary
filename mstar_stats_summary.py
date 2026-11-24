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
plateau start), or --disp-time; without an a(t) plateau from the start of the
window its kinetics fall back to. Per-time-point dispersion metrics (a, E = a/a0,
normalized a, integral of a dt, d32, Hinze d_max from eps(t)) go to
dispersion_timeseries.csv.
Without a plateau (mean velocity, a(t) or S(t)), averages and kinetics use the
last max(window, 25%) of the run and are labelled 'fallback' / 'no plateau'.
With --plot, the dispersion results are also plotted to stats_plots/dispersion.html.
The --dispersion / --uniformity metrics are also written on their own to
stats_summary_{dispersion,uniformity}.csv and, as one-row-per-case means with
bar plots, *_{dispersion,uniformity}_means.csv/.html (batch: batch_summary_*).
Simulation settings (dx, LX/LY/LZ, dt, Courant number, fluid volume, impeller
speeds, fluid model and properties, simulated and wall-clock time) from input.xml,
Fluid.txt and Timing.txt go to *_settings.csv.

Optional gas-liquid mass transfer through the free surface (M-Star free-surface
model only):
  --gas-liquid free-surface --gas-diffusivity D
                specific free-surface area a = A/V_liq and liquid-side kL, kLa of
                several models (Lamont-Scott, Kawase-Moo-Young, Kastanek, rigid
                surface, Higbie, Thomas, surface divergence): at every Stats sample
                from the bulk dissipation P/(rho V_liq) and the surface means of
                Interface.txt (gas_liquid_timeseries.csv), and per VOF surface
                frame from the fields of the Volume output at the surface,
                kLa = (1/V_liq) sum kL_i dA_i (gas_liquid_surface.csv; needs
                pyvista). gas_liquid_assessment.csv compares the voxel size with
                the one a calculation without a kL model would need (Kolmogorov
                scale and concentration film at the surface). Summary rows go to
                stats_summary_gas_liquid.csv (batch: batch_summary_gas_liquid*).

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
    python mstar_stats_summary.py case --gas-liquid free-surface --gas-diffusivity 2e-9

Cases:
- /data2/mstar/tak360/step8

"""

from __future__ import annotations

import os
import sys

# BLAS/OpenMP pools start one thread per core (128 here) and spin: cap them before numpy
# loads; an environment value wins. Skipped when numpy is already loaded (the M-Star Post
# app imports this file as a module): it would not reach numpy, only the app's child jobs.
if "numpy" not in sys.modules:
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, "4")

import argparse
import csv
import re
import shutil
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


def stats_rank(base: Path, stats_dir: Path) -> tuple:
    """Sort key of the Stats folders below a case folder: <base>/out/Stats (M-Star's
    layout) first, then <base>/Stats, then the shallowest, so that a copy such as
    backup/out/Stats never wins over the case's own."""
    rel = stats_dir.relative_to(base).parts
    return (0 if rel == ("out", "Stats") else 1 if rel == ("Stats",) else 2, len(rel), stats_dir)


def locate_stats_dir(root: Path) -> Path | None:
    """Accept a case dir, a dir containing Stats/, or the Stats dir itself. Of several
    Stats folders below a case dir the stats_rank() order decides."""
    if root.is_dir():
        if root.name.lower() == "stats":
            return root
        for d in (root / "out" / "Stats", root / "Stats"):
            if d.is_dir():
                return d
        hits = sorted((p for p in root.rglob("Stats") if p.is_dir() and not is_hidden(p, root)),
                      key=lambda p: stats_rank(root, p))
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


_RUN_SNAPSHOT = re.compile(r"^run\d+$", re.I)


def find_stats_files(stats_dir: Path) -> list[Path]:
    """Stats tables below stats_dir, without M-Star's restart snapshots (Stats/run1,
    run2, ...: the history up to each restart, which the top-level files hold in full)."""
    return sorted(f for f in stats_dir.rglob("*.txt")
                  if f.is_file() and not is_hidden(f, stats_dir)
                  and not any(_RUN_SNAPSHOT.match(p) for p in f.relative_to(stats_dir).parts[:-1]))


# ---------------------------------------------------------------- parsing

_ROWS_WARNED: set[str] = set()
_TABLES: dict[str, tuple] = {}  # path -> (mtime_ns, size, table); the newest TABLE_CACHE files
TABLE_CACHE = 32


def load_stats_table(path: Path) -> tuple[list[str], np.ndarray] | None:
    """Return (column headers, data[rows, cols]) or None if not a stats table.
    Rows whose column count differs from the header (a garbled line, the half-written
    last line of a running solve) are skipped with a warning; the rest is kept.
    A file is parsed once per (path, mtime, size): one case reads Fluid.txt eight times.
    The array is shared between callers and therefore read-only."""
    try:
        st = os.stat(path)
        key, stamp = str(path), (st.st_mtime_ns, st.st_size)
    except OSError:
        key = stamp = None
    if key in _TABLES and _TABLES[key][:2] == stamp:
        return _TABLES[key][2]
    table = _parse_stats_table(path)
    if key is not None:
        _TABLES.pop(key, None)
        _TABLES[key] = (*stamp, table)
        while len(_TABLES) > TABLE_CACHE:
            del _TABLES[next(iter(_TABLES))]
    return table


def _parse_stats_table(path: Path) -> tuple[list[str], np.ndarray] | None:
    where = f"{path.parent.name}/{path.name}"
    try:
        with open(path) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            if len(header) < 2 or not header[0].lower().startswith("time"):
                return None
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    except (OSError, UnicodeDecodeError) as e:
        warn(f"{where}: unreadable ({e})")
        return None
    good = [ln for ln in lines if ln.count("\t") == len(header) - 1]
    if len(good) < len(lines) and str(path) not in _ROWS_WARNED:
        _ROWS_WARNED.add(str(path))  # a file is read several times per case: say it once
        bad = [i + 2 for i, ln in enumerate(lines) if ln.count("\t") != len(header) - 1]
        warn(f"{where}: {len(bad)} of {len(lines)} row(s) skipped, column count differs from the "
             f"header (line {', '.join(map(str, bad[:5]))}{', ...' if len(bad) > 5 else ''})")
    if not good:
        warn(f"{where}: no data rows; skipped")
        return None
    try:
        data = np.genfromtxt(good, delimiter="\t")
    except Exception as e:
        warn(f"{where}: unreadable ({e})")
        return None
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] == 0 or data.shape[1] != len(header):
        warn(f"{where}: data shape does not match header; skipped")
        return None
    data.flags.writeable = False
    return header, data


def var_unit(name: str) -> str:
    m = re.search(r"\[([^\]]*)\]", name)
    return m.group(1) if m else ""


def norm_name(name: str) -> str:
    """Lowercase and strip the unit suffix, e.g. 'Mean Velocity [m/s]' -> 'mean velocity'."""
    return re.sub(r"\[[^\]]*\]", "", name).strip().lower()


# ---------------------------------------------------------------- steady state

EXCURSION_SHARE = 0.10  # largest share of unsteady samples a plateau may contain


def plateau_index(times: np.ndarray, steady: np.ndarray, window: float) -> int | None:
    """Index of the first sample of the plateau, given steady[i] = the window mean ending
    at sample i did not change against the window before it.

    Strictly, the plateau starts at the first steady sample after which every sample is
    steady. A single sample failing the test late in a fluctuating trace must not move
    the plateau to the end of the run, though, so short excursions are tolerated in a
    plateau long enough to judge them: runs of unsteady samples that last less than one
    window, making up at most EXCURSION_SHARE of the samples of a plateau of at least
    window / EXCURSION_SHARE. A run of a window or longer is a real change and the
    plateau starts after it. None if there is no plateau."""
    n = len(times)
    bad = ~steady
    edges = np.flatnonzero(np.diff(np.concatenate(([0], bad.astype(np.int8), [0]))))
    first, last = edges[::2], edges[1::2] - 1  # inclusive ends of each unsteady run
    real = times[last] - times[first] >= window
    i0 = int(last[real][-1]) + 1 if real.any() else 0
    share = np.cumsum(bad[::-1])[::-1] / (n - np.arange(n))
    ok = steady & (share <= EXCURSION_SHARE) & (times[-1] - times >= window / EXCURSION_SHARE)
    ok[:i0] = False
    ok |= steady & (share == 0)
    hit = np.flatnonzero(ok)
    return int(hit[0]) if hit.size else None


def _steady_at(i: int, cur, prev, times, values, window: float, tol: float, scale: float) -> bool:
    """The steadiness rule for sample i; cur / prev select the samples of the window
    ending at times[i] and of the window before it."""
    v_cur, v_prev = values[cur], values[prev]
    if times[i] < times[0] + 2 * window or len(v_cur) < 3 or len(v_prev) < 3:
        return False
    m_cur, m_prev = v_cur.mean(), v_prev.mean()
    denom = max(abs(m_prev), tol * scale)
    return bool(abs(m_cur - m_prev) / denom < tol)


def steady_flags_loop(times: np.ndarray, values: np.ndarray, window: float, tol: float,
                      scale: float) -> np.ndarray:
    """steady[i] = the mean over (t_i - window, t_i] differs by less than tol (relative to
    the earlier mean, at least tol * scale) from the mean over the window before it; both
    windows hold >= 3 samples and t_i is two windows into the trace. Reference rule,
    one mask pair per sample: O(n^2), any time order."""
    steady = np.zeros(len(times), bool)
    for i, t in enumerate(times):
        steady[i] = _steady_at(i, (times > t - window) & (times <= t),
                               (times > t - 2 * window) & (times <= t - window),
                               times, values, window, tol, scale)
    return steady


def steady_flags(times: np.ndarray, values: np.ndarray, window: float, tol: float,
                 scale: float) -> np.ndarray:
    """The flags of steady_flags_loop in O(n log n) for sorted times: the windows are
    index ranges (searchsorted) and their means come from prefix sums of the centred
    values. A sample whose drift is closer to tol than the round-off bound of those sums
    is decided by the reference rule on its two slices, so the flags are exactly those
    of the loop. Unsorted times or non-finite values use the loop."""
    n = len(times)
    if n < 3 or not (np.all(np.diff(times) >= 0) and np.isfinite(values).all()
                     and np.isfinite(scale)):
        return steady_flags_loop(times, values, window, tol, scale)
    hi = np.searchsorted(times, times, side="right")
    mid = np.searchsorted(times, times - window, side="right")
    lo = np.searchsorted(times, times - 2 * window, side="right")
    n_cur, n_prev = hi - mid, mid - lo
    ok = ~(times < times[0] + 2 * window) & (n_cur >= 3) & (n_prev >= 3)
    eps = np.finfo(float).eps
    c = float(values.mean())
    cs = np.concatenate(([0.0], np.cumsum(values - c)))
    err = eps * np.concatenate(([0.0], np.cumsum(np.abs(cs[1:]))))  # running error bound of cs
    with np.errstate(invalid="ignore", divide="ignore"):
        m_cur = (cs[hi] - cs[mid]) / n_cur
        m_prev = (cs[mid] - cs[lo]) / n_prev
        denom = np.maximum(np.abs(m_prev + c), tol * scale)
        drift = np.abs(m_cur - m_prev) / denom
        slack = 4 * ((err[hi] + err[mid]) / n_cur + (err[mid] + err[lo]) / n_prev
                     + 8 * eps * scale) / denom * (1 + drift) + 1e-13
    steady = ok & (drift < tol)
    for i in np.flatnonzero(ok & (np.abs(drift - tol) <= slack)):
        steady[i] = _steady_at(i, slice(mid[i], hi[i]), slice(lo[i], mid[i]),
                               times, values, window, tol, scale)
    return steady


def detect_steady_start(times: np.ndarray, values: np.ndarray,
                        window: float, tol: float) -> float | None:
    """First time from which the windowed mean stops drifting (change < tol).

    Compares the mean of the trailing window against the window before it, which
    tolerates turbulent fluctuations that never settle below tol themselves; short
    excursions inside the plateau are tolerated too (plateau_index).
    Returns None if no steady plateau is found.
    """
    if not values.size:
        return None
    scale = np.max(np.abs(values))
    if scale == 0.0:  # identically zero trace is steady from the start
        return float(times[0])

    steady = steady_flags(times, values, window, tol, scale)
    i = plateau_index(times, steady, window)
    # steady window [t-window, t] means the plateau began a window earlier
    return None if i is None else float(max(times[i] - window, times[0]))


FALLBACK = "fallback (not steady)"
NO_PLATEAU = "a(t) (no plateau, last window)"  # dispersion metrics when a(t) has no plateau


def fallback_start(times, window: float) -> float:
    """Start of the last max(window, 25% of the run): the averaging window
    used when no plateau is found."""
    times = np.asarray(times, float)
    t0, t1 = float(np.nanmin(times)), float(np.nanmax(times))
    return max(t0, t1 - max(window, 0.25 * (t1 - t0)))


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
        name = fix_unit(header[j].strip())
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
              what: str = "variables", labels: bool = False) -> None:
    """Write rows; with labels, the header shows ts_label(column) (name [unit])."""
    with open(out_path, "w", newline="") as fh:
        if labels:
            csv.writer(fh).writerow([ts_label(c) for c in columns])
        w = csv.DictWriter(fh, fieldnames=columns)
        if not labels:
            w.writeheader()
        w.writerows(rows)
    log(f"wrote {out_path} ({len(rows)} {what})")


# per-frame CSV column keys -> "name [unit]" headers; dimensionless ratios keep
# their dimensions (m/m, m^3/m^3, ...), counts are [count]
TS_LABELS = [
    (r"time_s", "time [s]"),
    (r"z_m", "z [m]"),
    (r"region_id", "region_id [id]"),
    (r"(n_\w+)", r"\1 [count]"),
    (r"a_1_m", "a [1/m]"),
    (r"E", "E = a/a0 [(1/m)/(1/m)]"),
    (r"a_norm", "a_norm = (a - a0)/(a_ss - a0) [(1/m)/(1/m)]"),
    (r"int_a_s_m", "int_a [s/m]"),
    (r"eps_W_kg", "eps [W/kg]"),
    (r"d32_interface_m", "d32_interface [m]"),
    (r"d_hinze_m", "d_hinze [m]"),
    (r"d_hinze_dx", "d_hinze/dx [m/m]"),
    (r"(A_\w+?|area)_m2", r"\1 [m^2]"),
    (r"(V_\w+?|volume)_m3(_\w+)?", r"\1\2 [m^3]"),
    (r"(d32|d10|d43|d_max|d_eq)_m(_\w+)?", r"\1\2 [m]"),
    (r"d32_dx(_\w+)?", r"d32/dx\1 [m/m]"),
    (r"dmax_d32(_\w+)?", r"d_max/d32\1 [m/m]"),
    (r"coverage", "coverage A_drops/A_int [m^2/m^2]"),
    (r"((?:F_disp|phi_mean|share|cov|S)(?:_\w+)?)", r"\1 [m^3/m^3]"),
]


def ts_label(key: str) -> str:
    for pat, rep in TS_LABELS:
        if re.fullmatch(pat, key):
            return re.sub(pat, rep, key)
    return key


# summary variables written with "[-]" before units were made explicit
UNIT_FIXES = [
    (r"Interface Enhancement E = a_ss/a0", "(1/m)/(1/m)"),
    (r"Interfacial Area Plateau CoV", "(1/m)/(1/m)"),
    (r"Number of Drops n_drops \([^)]*\)", "count"),
    (r"Sauter Diameter d32 / Voxel Size \([^)]*\)", "m/m"),
    (r"Dispersed Fraction F_disp \([^)]*\)", "m^3/m^3"),
    (r"Drop Interface Coverage A_drops/A_int", "m^2/m^2"),
    (r"Zonal CoV of Phase Fraction \([^)]*\)", "m^3/m^3"),
    (r"Segregation Index S \([^)]*\)", "m^3/m^3"),
    (r"Largest Region Share \([^)]*\)", "m^3/m^3"),
    (r"Hinze Drop Size / Voxel Size", "m/m"),
    (r"\bL[XYZ]", "cells"),
    (r"(?:Max )?Courant number \([^)]*\)", "m/m"),
    # M-Star Stats variables labelled [-]
    (r"\bIteration", "count"),
    (r"LB Density (?:Max|Min)", "(kg/m^3)/(kg/m^3)"),
    (r"Power Number(?: Normalized by [^\[]*?)?", "W/W"),
]


def fix_unit(name: str) -> str:
    for pat, unit in UNIT_FIXES:
        name = re.sub(f"({pat}) \\[-\\]", lambda m: f"{m.group(1)} [{unit}]", name)
    return name


def migrate_units(root: Path, ref_case: str | None = None) -> int:
    """Rewrite existing result CSVs under root with the current unit labels
    (time-series headers and summary variable names/units); regenerates the
    bar-plot HTML of changed *_means.csv. Returns the number of files changed."""
    changed = 0
    for f in sorted(root.rglob("*.csv")):
        with open(f, newline="") as fh:
            rows = list(csv.reader(fh))
        if not rows or not rows[0]:
            continue
        hdr = rows[0]
        new = [r[:] for r in rows]
        if hdr[0] == "time_s":
            new[0] = [ts_label(c) for c in hdr]
        else:
            new[0] = [fix_unit(c) for c in hdr]
            if "variable" in hdr and "unit" in hdr:
                iv, iu = hdr.index("variable"), hdr.index("unit")
                for r in new[1:]:
                    if len(r) > max(iv, iu) and fix_unit(r[iv]) != r[iv]:
                        r[iv] = fix_unit(r[iv])
                        r[iu] = var_unit(r[iv])
        if new == rows:
            continue
        with open(f, "w", newline="") as fh:
            csv.writer(fh).writerows(new)
        changed += 1
        log(f"updated units in {f}")
        html = f.with_suffix(".html")
        if f.stem.endswith("_means") and html.is_file() and new[0][0] == "case":
            def num(v):
                try:
                    return float(v)
                except ValueError:
                    return v
            data = {r[0]: {c: num(v) for c, v in zip(new[0][1:], r[1:]) if v != ""}
                    for r in new[1:] if r}
            plot_means_bars(new[0][1:], data, html, ref_case)
    return changed


# M-Star's default moving-body name "Moving Body" and a body named "agitator"
# are the same impeller; results are normalised to one canonical name so cases
# share columns ("MovingBody_Agitator: ...", "Impeller Power P (Agitator) [W]").
BODY_CANON = "Agitator"
_BODY_ALIASES = re.compile(r"Moving[ _]Body|(?<![A-Za-z])agitator(?![A-Za-z])")


def canon_body(text: str, canon: str = BODY_CANON) -> str:
    """text with every alias of the impeller body name replaced by canon."""
    return _BODY_ALIASES.sub(canon, text)


def _merge_columns(rows: list[list[str]], renamed: set[int]) -> list[list[str]]:
    """Merge the columns whose header the rename made equal to another one (renamed =
    indices of the headers it changed): per row the first non-empty value wins (cases
    that used different body names). Headers that were equal before are left alone."""
    hdr = rows[0]
    groups: dict[str, list[int]] = {}
    for i, h in enumerate(hdr):
        groups.setdefault(h, []).append(i)
    dups = {g[0]: g[1:] for g in groups.values() if len(g) > 1 and renamed.intersection(g)}
    if not dups:
        return rows
    drop = {j for g in dups.values() for j in g}
    keep = [i for i in range(len(hdr)) if i not in drop]
    out = [[hdr[i] for i in keep]]
    for r in rows[1:]:
        r = r + [""] * (len(hdr) - len(r))
        out.append([next((r[j] for j in [i, *dups.get(i, [])] if r[j] != ""), "") for i in keep])
    return out


BODY_CELLS = ("file", "variable")  # columns whose cells name the body (never paths or case ids)


def migrate_body_names(root: Path, ref_case: str | None = None,
                       canon: str = BODY_CANON) -> int:
    """Rename the impeller body in existing results under root to canon: CSV headers
    and the cells of the file / variable columns (merging columns the rename made
    identical), HTML text and file names; regenerates the bar plot of changed
    *_means.csv. Other cells (case ids, the input.xml path, notes) are left as they
    are. Idempotent. Returns the number of files changed."""
    changed = 0
    for f in sorted(root.rglob("*.csv")):
        with open(f, newline="") as fh:
            rows = list(csv.reader(fh))
        if not rows or not rows[0]:
            continue
        hdr = [canon_body(h, canon) for h in rows[0]]
        cells = {j for j, h in enumerate(rows[0]) if h in BODY_CELLS}
        new = _merge_columns([hdr] + [[canon_body(v, canon) if j in cells else v for j, v in enumerate(r)]
                                      for r in rows[1:]],
                             {j for j, h in enumerate(hdr) if h != rows[0][j]})
        if new == rows:
            continue
        with open(f, "w", newline="") as fh:
            csv.writer(fh).writerows(new)
        changed += 1
        log(f"renamed body in {f}")
        html = f.with_suffix(".html")
        if f.stem.endswith("_means") and html.is_file() and new[0][0] == "case":
            def num(v):
                try:
                    return float(v)
                except ValueError:
                    return v
            data = {r[0]: {c: num(v) for c, v in zip(new[0][1:], r[1:]) if v != ""}
                    for r in new[1:] if r}
            plot_means_bars(new[0][1:], data, html, ref_case)
    for f in sorted(root.rglob("*.html")):
        if f.stem.endswith("_means"):
            continue  # regenerated above from the CSV
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new = canon_body(text, canon)
        if new != text:
            f.write_text(new, encoding="utf-8")
            changed += 1
            log(f"renamed body in {f}")
    for f in sorted(root.rglob("*")):
        if f.is_file() and canon_body(f.name, canon) != f.name:
            target = f.with_name(canon_body(f.name, canon))
            if not target.exists():
                f.rename(target)
                changed += 1
                log(f"renamed {f.name} -> {target.name}")
    return changed


DEFAULT_MEANS_VARS = [
    "energy dissipation", "velocity", "vorticity", "mixed time", "shear",
    "power number", "fluid volume", "viscosity", "density",
    "rotation speed", "angular velocity",
    "interfacial area", "d32", "dispersed fraction", "n_drops", "cov",
    "coverage", "hinze", "interface enhancement", "dispersion time",
    "time-integrated", "segregation", "homogeneity time", "continuous phase",
    "impeller power", "power per volume", "maximum dissipation", "kolmogorov",
    "blend time", "d/d32", "d_max/d32",
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

    if HTML_PLOTS:
        plot_means_bars(cols, data, out_path.with_suffix(".html"), ref_case)


# HTML figures (means bar plots, per-file Stats plots, dispersion.html) are legacy: the
# M-Star Post app draws everything from the CSVs. --plot-html turns them back on.
HTML_PLOTS = False


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
    """Steady-state (t >= t_start) summary row of a derived time series; if no
    sample is at t >= t_start, the last 25% of the samples (labelled fallback)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    finite = np.isfinite(t) & np.isfinite(v)
    ok = finite & (t >= t_start)
    if not ok.any():
        if not finite.any():
            warn(f"  {file}: no finite '{variable}' data")
            return None
        t_fb = fallback_start(t[finite], 0.0)
        warn(f"  {file}: no finite '{variable}' data at t >= {t_start:g} s; "
             f"averaging the last samples t >= {t_fb:g} s instead")
        ok = finite & (t >= t_fb)
        method, t_start = f"{method} (fallback, before steady start)", t_fb
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
                      levels=(0.90, 0.95), fallback: bool = False) -> dict | None:
    """Approach of y(t) from its first sample y0 to its plateau y_ss.

    Returns y0, plateau start t_ss (detect_steady_start), y_ss = mean over the
    plateau, plateau_cov = std/y_ss there, t<level> = first time
    frac = (y - y0)/(y_ss - y0) reaches each level (linear interpolation) and
    tau from a log-linear fit of ln(1 - frac) vs t over 0.05 < frac < 0.95.
    Plateau-dependent values are NaN if y never plateaus, unless fallback: then
    y_ss is the mean over the last window (fallback_start) and plateau = False;
    t_ref is the start of the averaging window. None if < 2 samples.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size < 2:
        return None
    keys = [f"t{100 * lev:g}" for lev in levels]
    out = {"y0": float(y[0]), "t_ss": np.nan, "y_ss": np.nan, "plateau_cov": np.nan,
           "tau": np.nan, "plateau": False, "t_ref": np.nan, **{k: np.nan for k in keys}}
    t_ss = detect_steady_start(t, y, window, tol)
    if t_ss is not None:
        out.update(t_ss=t_ss, plateau=True)
    elif fallback:
        t_ss = fallback_start(t, window)
    else:
        return out
    plateau = t >= t_ss
    y_ss = float(y[plateau].mean())
    out.update(t_ref=t_ss, y_ss=y_ss)
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


def dispersion_kinetics(t, a, window: float, tol: float, levels=(0.90, 0.95),
                        fallback: bool = False) -> dict:
    """Dispersion kinetics of the specific interfacial area a(t): a0 (flat
    interface at impeller start), plateau start t_ss_a, a_ss, enhancement
    E = a_ss/a0, t90/t95, time constant tau, plateau CoV and int_a = integral
    of a dt from t = 0. Empty dict if a(t) has < 2 samples. With fallback,
    a_ss is the last-window mean when a(t) never plateaus (plateau = False)."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    ok = np.isfinite(t) & np.isfinite(a)
    t, a = t[ok], a[ok]
    k = approach_kinetics(t, a, window, tol, levels, fallback)
    if k is None:
        return {}
    a0 = k["y0"]
    if t[0] > 0:  # a is the flat-interface a0 until the first sample
        t, a = np.r_[0.0, t], np.r_[a0, a]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    out = {"a0": a0, "t_ss_a": k["t_ss"], "a_ss": k["y_ss"],
           "E": k["y_ss"] / a0 if a0 > 0 else np.nan, "tau": k["tau"],
           "plateau_cov": k["plateau_cov"], "int_a": float(trapz(a, t)),
           "plateau": k["plateau"], "t_ref": k["t_ref"]}
    out.update({f"t{100 * lev:g}": k[f"t{100 * lev:g}"] for lev in levels})
    return out


KINETICS_VARS = [  # (dispersion_kinetics key, summary variable)
    ("a0", "Initial Specific Interfacial Area a0 [1/m]"),
    ("a_ss", "Plateau Specific Interfacial Area a_ss [1/m]"),
    ("A_ss", "Plateau Interfacial Area A_ss = a_ss V_liq [m^2]"),
    ("t_ss_a", "Interfacial Area Plateau Start t_ss_a [s]"),
    ("E", "Interface Enhancement E = a_ss/a0 [(1/m)/(1/m)]"),
    ("t90", "Dispersion Time t90 [s]"),
    ("t95", "Dispersion Time t95 [s]"),
    ("tau", "Dispersion Time Constant tau [s]"),
    ("plateau_cov", "Interfacial Area Plateau CoV [(1/m)/(1/m)]"),
    ("int_a", "Time-Integrated Specific Interfacial Area [s/m]"),
]


def interface_table(files: list[Path], stats_dir: Path, area_column: str | None = None,
                    flag: str = "--dispersion") -> tuple[str, list[str], np.ndarray, int] | None:
    """(file name relative to stats_dir, header, data, index of the area column) of the
    interface stats file (M-Star's Interface.txt: the VOF surface area and the surface
    means of the fields written with it); None with a warning naming flag if unusable."""
    hits = [f for f in files if "interface" in f.name.lower()]
    if not hits:
        warn(f"{flag}: no interface stats file (name containing "
             "'interface') found; skipping interfacial-area metrics")
        return None
    f = hits[0]
    rel = str(f.relative_to(stats_dir))
    table = load_stats_table(f)
    if table is None:
        warn(f"{flag}: {rel} is not a stats table; skipping interfacial-area metrics")
        return None
    header, data = table
    normed = [norm_name(h) for h in header]
    cols = [j for j in range(1, len(header)) if "area" in normed[j]]
    if area_column:
        q = area_column.strip().lower()
        cols = [j for j in range(1, len(header)) if q in normed[j]]
        if not cols:
            warn(f"{flag}: --area-column '{q}' matched no column in {rel}; "
                 "available: " + ", ".join(h.strip() for h in header[1:]))
            return None
    elif len(cols) > 1:
        warn(f"{flag}: several area columns in {rel}: "
             + ", ".join(f"'{header[j].strip()}'" for j in cols)
             + f"; using '{header[cols[0]].strip()}' (choose with --area-column)")
    if not cols:
        warn(f"{flag}: no interfacial area column in {rel}; available: "
             + ", ".join(h.strip() for h in header[1:]))
        return None
    log(f"interfacial area: '{header[cols[0]].strip()}' from {rel}")
    return rel, header, data, cols[0]


def interface_metrics(files: list[Path], stats_dir: Path, t_start: float,
                      phi: float | None, v_liq: float | None, window: float = 1.0,
                      tol: float = 0.02, vessel_diameter: float | None = None,
                      area_column: str | None = None) -> tuple[list[dict], dict]:
    """Specific interfacial area a = A_int/V_liq, area-based d32 = 6*phi/a and
    the dispersion kinetics of a(t) (dispersion_kinetics()).

    Returns (summary rows, {"t", "a", "area"[, "d32"], "kinetics"} time series).
    """
    found = interface_table(files, stats_dir, area_column)
    if found is None:
        return [], {}
    rel, header, data, j = found
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
    series = {"t": t, "a": a, "area": area, "vol": vol}
    rows = [series_row(rel, "Specific Interfacial Area [1/m]", t, a, t_start, "interface"),
            series_row(rel, "Interfacial Area A_int [m^2]", t, area, t_start, "interface")]

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

    kin = dispersion_kinetics(t, a, window, tol, fallback=True)
    series["kinetics"] = kin
    if not kin:
        return rows, series
    tf = t[np.isfinite(t) & np.isfinite(a)]
    if tf.size > 1 and tf[0] > np.median(np.diff(tf)):
        warn(f"--dispersion: first a(t) sample at t = {tf[0]:g} s, more than one "
             "output interval after 0; a0 may not be the flat interface")
    if not kin["plateau"]:
        warn("--dispersion: a(t) never plateaus: dispersion still developing; "
             "extend the run (kinetics below use the last-window mean of a, "
             f"t >= {kin['t_ref']:.4g} s, as a_ss)")
    if vessel_diameter:
        v0 = vol[np.isfinite(t) & np.isfinite(a)][0]
        a_flat = np.pi * vessel_diameter ** 2 / 4.0 / v0
        dev = kin["a0"] / a_flat - 1.0
        (warn if abs(dev) > 0.25 else log)(
            f"  a0 = {kin['a0']:.4g} 1/m vs flat-interface estimate "
            f"pi*D^2/4/V_liq = {a_flat:.4g} 1/m ({100 * dev:+.0f}%)")
    t_ss = kin["t_ss_a"] if kin["plateau"] else kin["t_ref"]
    v_pl = vol[np.isfinite(t) & np.isfinite(vol) & (t >= t_ss)]
    kin["A_ss"] = kin["a_ss"] * float(v_pl.mean()) if v_pl.size else np.nan
    method = "a(t) kinetics" if kin["plateau"] else "a(t) kinetics (no plateau, last window)"
    for key, var in KINETICS_VARS:
        v = kin.get(key, np.nan)
        if np.isfinite(v):
            rows.append(value_row(rel, var, float(v), method, t_ss,
                                  float(tf[-1]), int(tf.size)))
    log("  a(t) kinetics: " + ", ".join(f"{k} = {kin.get(k, np.nan):.4g}" for k, _ in KINETICS_VARS))
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


def vtk_complete(path: Path) -> bool:
    """False for a VTK XML file that is still being written: it lacks the closing tag.
    Such a file reads without an error as an empty or partial mesh."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 512))
            return b"</VTKFile>" in fh.read()
    except OSError:
        return False


def complete_frames(frames: list[tuple[float | None, list[Path]]],
                    what: str) -> list[tuple[float | None, list[Path]]]:
    """frames without the ones of a running solve's current output step: a file still
    being written, or a time step that not every block has written yet."""
    n_blocks: dict[Path, int] = {}
    for t, paths in frames:
        if t is not None:
            n_blocks[paths[0].parent.parent] = max(n_blocks.get(paths[0].parent.parent, 0), len(paths))
    keep, dropped = [], []
    for t, paths in frames:
        whole = t is None or len(paths) == n_blocks[paths[0].parent.parent]
        (keep if whole and all(vtk_complete(p) for p in paths) else dropped).append((t, paths))
    if dropped:
        names = ", ".join(paths[0].name for _, paths in dropped[:3])
        warn(f"  {what}: {len(dropped)} incomplete frame(s) skipped (still being written, or "
             f"missing in one block): {names}{', ...' if len(dropped) > 3 else ''}")
    return keep


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


# <fluidModelType> of input.xml (as written by M-Star 3.x for the cases on this server)
FLUID_MODELS = {1: "single fluid", 2: "free surface", 4: "immiscible two-fluid"}
FREE_SURFACE = FLUID_MODELS[2]


def fluid_model(fm) -> str | None:
    """Name of the M-Star fluid model of a <fluidModel> element ('free surface', ...);
    'type N' for a type this script does not know, None without the tag."""
    txt = fm.findtext("fluidModelType") if fm is not None else None
    try:
        n = int(float(txt))
    except (TypeError, ValueError):
        return None
    return FLUID_MODELS.get(n, f"type {n}")


def case_settings(case: Path, stats_dir: Path, case_id: str | None = None) -> dict:
    """Simulation settings of one case from input.xml (dx, LX/LY/LZ, dt, Courant
    number, runtime, impeller speeds, fluid model, per-fluid density/viscosity, surface tension),
    Fluid.txt (initial fluid volume, max simulated Courant number, simulated end
    time) and Timing.txt (iterations, wall time)."""
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
    s: dict = {"case": case_id or case.name}
    dx = dt = None
    imps: list[tuple[str, object, object]] = []  # (name, speed [rpm], diameter [m])
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
                s[f"L{a} [cells]"] = int(round(length / dx))
        s["dt [s]"] = dt
        s["Simulated time (input runtime) [s]"] = num(sys_el, "runtime")
        # M-Star sets dt = uMax*dx/U_ref, so uMax is the Courant number at U_ref
        s["Courant number (input uMax) [m/m]"] = num(sys_el, "uMax")
        for imp in root.findall("impeller"):
            name = (imp.findtext("name") or "impeller").strip()
            s[f"Impeller speed ({name}) [rpm]"] = num(imp, "freq")
            s[f"Impeller diameter ({name}) [m]"] = num(imp, "diameter")
            imps.append((name, num(imp, "freq"), num(imp, "diameter")))
        fm = root.find("fluidModel")
        if fm is not None:
            s["Fluid model"] = fluid_model(fm)
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
        t = data[:, 0][np.isfinite(data[:, 0])]
        if t.size:
            s["Simulated time (end) [s]"] = float(t[-1])
        if "fluid volume" in normed:
            v = data[:, normed.index("fluid volume")]
            v = v[np.isfinite(v)]
            if v.size:
                s["Fluid volume (total, initial) [m^3]"] = float(v[0])
        if "max velocity" in normed and isinstance(dx, float) and isinstance(dt, float):
            u = data[:, normed.index("max velocity")]
            if np.isfinite(u).any():
                s["Max Courant number (simulated, max over run) [m/m]"] = \
                    float(np.nanmax(u)) * dt / dx
    s.update(run_timing(stats_dir))
    if root is not None:
        s.update(tank_geometry(root, s.get("Fluid volume (total, initial) [m^3]")))
    s.update(dimensionless_groups(s, imps, fluid_volumes(find_stats_files(stats_dir))))
    return {k: v for k, v in s.items() if v is not None}


_TIMING_WARNED: set[str] = set()


def run_timing(stats_dir: Path) -> dict:
    """Iterations and wall-clock time from the M-Star Timing.txt. The cumulative wall
    time counter normally runs on over restarts (last row = total); where it starts
    again (drops to less than half), the runs are summed. Rows of several solver runs
    mixed in one file (time going backwards) cannot be told apart: then the largest
    counter value is taken, with a warning."""
    f = next((f for f in find_stats_files(stats_dir) if f.name.lower() == "timing.txt"), None)
    table = load_stats_table(f) if f else None
    if table is None:
        return {}
    header, data = table
    normed = [norm_name(h) for h in header]
    out: dict = {}
    t = data[:, 0]
    mixed = int((np.diff(t[np.isfinite(t)]) < 0).sum())
    notes = [f"time goes backwards {mixed} time(s), rows of several solver runs are mixed: "
             "iterations and wall time are the largest counter values, not sums over "
             "restarts"] if mixed else []
    if "iteration" in normed:
        it = data[:, normed.index("iteration")]
        it = it[np.isfinite(it)]
        if it.size:
            out["Iterations [count]"] = int(it.max() if mixed else it[-1])
    if "cumulative wall time" in normed:
        w = data[:, normed.index("cumulative wall time")]
        w = w[np.isfinite(w)]
        if w.size:
            drops = np.flatnonzero(np.diff(w) < 0)
            restarts = drops[w[drops + 1] < 0.5 * w[drops]]
            if mixed:
                minutes = float(w.max())
            else:  # one run per restart of the counter; other backward steps are not added
                minutes = float(sum(run.max() for run in np.split(w, restarts + 1)))
                if len(drops) > len(restarts):
                    notes.append(f"the wall time counter steps back {len(drops) - len(restarts)} "
                                 "time(s) without starting again; those steps are ignored")
            out["Wall time (cumulative) [h]"] = minutes / 60.0
    if notes and str(f) not in _TIMING_WARNED:
        _TIMING_WARNED.add(str(f))  # read twice per case
        warn(f"{f.parent.name}/{f.name}: " + "; ".join(notes))
    return out


def speed_mag(rpm):
    """|speed|: M-Star's <freq> is signed (the sense of rotation)."""
    return abs(rpm) if isinstance(rpm, (int, float)) else rpm


def impeller_info(root) -> list[dict]:
    """Impellers of an input.xml: name, speed [rpm], diameter [m], unit rotation axis."""
    def f(el, tag):
        try:
            return float(el.findtext(tag))
        except (TypeError, ValueError):
            return None
    out = []
    for imp in root.findall("impeller"):
        try:
            ax = np.array([float(x) for x in (imp.findtext("rotationAxis") or "").split()])
        except ValueError:
            ax = np.zeros(0)
        ax = ax / np.linalg.norm(ax) if ax.size == 3 and np.linalg.norm(ax) > 0 else None
        out.append({"name": (imp.findtext("name") or "impeller").strip(),
                    "rpm": f(imp, "freq"), "D": f(imp, "diameter"), "axis": ax})
    return out


def tank_geometry(root, v_liq: float | None) -> dict:
    """Tank diameter T = smaller horizontal domain extent (normal to the first
    impeller axis, default y) and liquid height H = V_liq / (pi T^2 / 4)."""
    sys_el = root.find("system")
    try:
        dom = [float(sys_el.findtext(f"domain{a}")) for a in "XYZ"]
    except (AttributeError, TypeError, ValueError):
        return {}
    imps = [i for i in impeller_info(root) if i["axis"] is not None]
    vert = int(np.argmax(np.abs(imps[0]["axis"]))) if imps else 1
    t = min(dom[i] for i in range(3) if i != vert)
    out = {"Tank diameter T (domain width) [m]": t}
    if v_liq and t > 0:
        out["Liquid height H = V/(pi T^2/4) [m]"] = v_liq / (np.pi * t ** 2 / 4)
    return out


D32_C1, D32_C3 = 0.053, 3.0  # HIM eq. (12-30), p. 664: Chen & Middleman C, Doulah b


def dimensionless_groups(s: dict, imps: list[tuple], fluids: list[tuple]) -> dict:
    """Per impeller: D/T, tip speed, Re, Fr and, for two fluids, We, the viscosity
    group Vi and the d32 correlations of HIM Ch. 12: eq. (12-30)
    d32 = D C1 (1 + C3 phi) We^-0.6 (inviscid drops, noncoalescing, phi < 0.3) and
    eq. (12-28) d32 = D 0.053 We^-0.6 (1 + 0.92 Vi^0.84)^0.6 (viscous drops, dilute).
    The continuous phase is the fluid with the larger initial volume."""
    out: dict = {}
    vols = {i: float(v[np.isfinite(v)][0]) for i, _, _, v in fluids if np.isfinite(v).any()}
    ic = max(vols, key=vols.get) if vols else 1
    idp = next((i for i in vols if i != ic), None)
    rho_c = s.get(f"Fluid {ic} density [kg/m^3]")
    nu_c = s.get(f"Fluid {ic} kinematic viscosity [m^2/s]")
    rho_d = s.get(f"Fluid {idp} density [kg/m^3]")
    mu_d = s.get(f"Fluid {idp} dynamic viscosity [Pa s]")
    sigma = s.get("Surface tension [N/m]")
    tank = s.get("Tank diameter T (domain width) [m]")
    phi = None
    if len(vols) >= 2 and sum(vols.values()) > 0:
        phi = 1.0 - vols[ic] / sum(vols.values())
        name = s.get(f"Fluid {ic} name") or ""
        out["Continuous phase (largest initial volume)"] = f"Fluid {ic} ({name})" if name else f"Fluid {ic}"
        out["Dispersed fraction phi (initial) [m^3/m^3]"] = phi
    isnum = lambda x: isinstance(x, float) and x > 0  # noqa: E731
    for name, rpm, d in imps:
        rpm = speed_mag(rpm)
        if not (isnum(rpm) and isnum(d)):
            continue
        n = rpm / 60.0
        if isnum(tank):
            out[f"D/T ({name}) [m/m]"] = d / tank
        out[f"Tip speed ({name}) [m/s]"] = np.pi * n * d
        if isnum(nu_c):
            out[f"Reynolds number Re ({name}) [N/N]"] = n * d ** 2 / nu_c
        out[f"Froude number Fr ({name}) [(m/s^2)/(m/s^2)]"] = n ** 2 * d / 9.81
        if isnum(rho_c) and isnum(sigma):
            we = rho_c * n ** 2 * d ** 3 / sigma
            out[f"Weber number We ({name}) [N/N]"] = we
            if phi is not None:
                out[f"d32 correlation ({name}) [m]"] = \
                    d * D32_C1 * (1 + D32_C3 * phi) * we ** -0.6
            if isnum(rho_d) and isnum(mu_d):
                vi = (rho_c / rho_d) ** 0.5 * mu_d * n * d / sigma
                out[f"Viscosity group Vi ({name}) [N/N]"] = vi
                out[f"d32 correlation, viscous drops ({name}) [m]"] = \
                    d * 0.053 * we ** -0.6 * (1 + 0.92 * vi ** 0.84) ** 0.6
    return out


def power_series(files: list[Path], imp: dict) -> tuple[Path, np.ndarray, np.ndarray] | None:
    """(Stats file, t, P(t)) of one impeller_info() entry: P = |torque . axis| |omega| from
    its MovingBody_<name>.txt; None without the file, the rotation axis or the columns."""
    name, ax = imp["name"], imp["axis"]
    f = next((f for f in files if f.name.lower() == f"movingbody_{name}.txt".lower()), None)
    table = load_stats_table(f) if f is not None and ax is not None else None
    if table is None:
        return None
    header, data = table
    normed = [norm_name(x) for x in header]
    need = ["torque x", "torque y", "torque z", "angular velocity"]
    if not all(k in normed for k in need):
        return None
    tq = np.stack([data[:, normed.index(k)] for k in need[:3]], axis=1)
    return f, data[:, 0], np.abs(tq @ ax) * np.abs(data[:, normed.index("angular velocity")])


def power_metrics(base: Path, files: list[Path], rows: list[dict], s: dict,
                  t_start: float) -> list[dict]:
    """Steady impeller power from the M-Star torque, P = |torque . axis| |omega|,
    and derived scales (HIM): P/V, power-based eps_avg = P/M and the resolved/power
    dissipation ratio, eps_max ~ P/(rho_c D^3) (power per impeller swept volume,
    eq. 2-19), Kolmogorov scale eta = (nu_c^3/eps)^(1/4) (eq. 12-15) at eps_avg
    and eps_max, Hinze d_max at eps_max (eq. 12-21), power number from torque
    and the turbulent blend-time correlation N t95 Po^(1/3) (D/T)^2 (T/H)^0.5 = 5.20
    (eq. 9-1), valid for Po^(1/3) Re > 6370."""
    p = find_input_xml(base)
    if p is None:
        return []
    import xml.etree.ElementTree as ET
    try:
        imps = impeller_info(ET.parse(p).getroot())
    except ET.ParseError:
        return []
    cont = re.match(r"Fluid (\d+)", str(s.get("Continuous phase (largest initial volume)", "")))
    ic = int(cont.group(1)) if cont else 1
    rho_c = s.get(f"Fluid {ic} density [kg/m^3]")
    nu_c = s.get(f"Fluid {ic} kinematic viscosity [m^2/s]")
    sigma, dx = s.get("Surface tension [N/m]"), s.get("dx [m]")
    v_liq = s.get("Fluid volume (total, initial) [m^3]")
    t_tank, h = s.get("Tank diameter T (domain width) [m]"), s.get("Liquid height H = V/(pi T^2/4) [m]")
    mass = sum((s.get(f"Fluid {i} density [kg/m^3]") or 0) * float(v[np.isfinite(v)][0])
               for i, _, _, v in fluid_volumes(files) if np.isfinite(v).any()) \
        or ((rho_c or 0) * (v_liq or 0))
    edr = next((r["mean"] for r in rows if r["file"].lower() == "fluid.txt"
                and norm_name(r["variable"]) == "mean energy dissipation rate"), None)
    pos = lambda x: isinstance(x, (int, float)) and np.isfinite(x) and x > 0  # noqa: E731
    out, p_total = [], 0.0

    def add(var, val, note="derived (steady means)"):
        if pos(val):
            out.append(value_row("derived", var, float(val), note, t_start))
            log(f"  {var}: {val:.4g}")

    if pos(edr) and pos(nu_c):
        add("Kolmogorov Scale at Mean Resolved Dissipation [m]", (nu_c ** 3 / edr) ** 0.25)
        if pos(dx):
            add("Kolmogorov Scale at Mean Resolved Dissipation / Voxel Size [m/m]",
                (nu_c ** 3 / edr) ** 0.25 / dx)
    for imp in imps:
        name, d, rpm = imp["name"], imp["D"], speed_mag(imp["rpm"])
        ps = power_series(files, imp)
        if ps is None:
            continue
        f, t_pow, power = ps
        row = series_row(f.name, f"Impeller Power P ({name}) [W]", t_pow, power, t_start,
                         "torque x omega")
        if row is None or not pos(row["mean"]):
            continue
        out.append(row)
        pw = row["mean"]
        p_total += pw
        log(f"  impeller power ({name}): {pw:.4g} W")
        if pos(d) and pos(rho_c):
            eps_max = pw / (rho_c * d ** 3)
            add(f"Maximum Dissipation Estimate P/(rho_c D^3) ({name}) [W/kg]", eps_max)
            if pos(nu_c):
                add(f"Kolmogorov Scale at Maximum Dissipation ({name}) [m]", (nu_c ** 3 / eps_max) ** 0.25)
            if pos(sigma):
                dmax = hinze_dmax(eps_max, sigma, rho_c)
                add(f"Hinze d_max at Maximum Dissipation ({name}) [m]", dmax)
                if pos(dx):
                    add(f"Hinze d_max at Maximum Dissipation / Voxel Size ({name}) [m/m]", dmax / dx)
            if pos(rpm):
                n = rpm / 60.0
                po = pw / (rho_c * n ** 3 * d ** 5)
                add(f"Power Number from Torque P/(rho_c N^3 D^5) ({name}) [W/W]", po)
                if pos(t_tank) and pos(h):
                    add(f"Blend Time t95 Correlation HIM eq. 9-1 ({name}) [s]",
                        5.20 * t_tank ** 1.5 * h ** 0.5 / (po ** (1 / 3) * n * d ** 2))
                if pos(nu_c):
                    add(f"Po^(1/3) Re ({name}) [N/N]", po ** (1 / 3) * n * d ** 2 / nu_c)
                    if po ** (1 / 3) * n * d ** 2 / nu_c < 6370:
                        warn(f"  {name}: Po^(1/3) Re < 6370, blend-time correlation eq. 9-1 "
                             "(turbulent) does not apply")
    if p_total > 0:
        if pos(v_liq):
            add("Power per Volume P/V (all impellers) [W/m^3]", p_total / v_liq)
        if pos(mass):
            add("Power-based Mean Dissipation P/M [W/kg]", p_total / mass)
            if pos(edr):
                add("Resolved / Power-based Dissipation [(W/kg)/(W/kg)]", edr / (p_total / mass))
    return out


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
                       np.asarray(mesh.cell_data[name]).reshape(dims[::-1]), mesh))
    if not blocks:
        return None

    def sample(pts: np.ndarray) -> np.ndarray:
        out = np.full(len(pts), np.nan)
        for origin, spacing, dims, arr, _ in blocks:
            ijk = np.floor((pts - origin) / spacing).astype(int)
            ok = np.all((ijk >= 0) & (ijk < dims), axis=1) & np.isnan(out)
            i, j, k = ijk[ok].T
            out[ok] = arr[k, j, i]
        return out
    return sample


def merge_points(raw: np.ndarray, step: float | None) -> tuple[np.ndarray, np.ndarray]:
    """(first, merge): index of the first point of every group of coinciding points
    (rounded to step if given), sorted by x, y, z, and the group of each point. The
    integer coordinates are packed into one int64 key (one sort instead of a row sort);
    np.unique(axis=0) gives the same and is used where the key would overflow."""
    if step:
        q = np.round(raw / step).astype(np.int64)
        q -= q.min(axis=0)
        span = [int(s) + 1 for s in q.max(axis=0)]
        if span[0] * span[1] * span[2] < 2 ** 62:
            _, first, merge = np.unique((q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2],
                                        return_index=True, return_inverse=True)
        else:
            _, first, merge = np.unique(q, axis=0, return_index=True, return_inverse=True)
    else:
        _, first, merge = np.unique(raw, axis=0, return_index=True, return_inverse=True)
    return first, merge.ravel()


def open_edges(F: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Edges that belong to one triangle only: (index of their triangle, [a, b] points)."""
    edges = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1).astype(np.int64)
    _, inv, cnt = np.unique(edges[:, 0] * n_points + edges[:, 1], return_inverse=True,
                            return_counts=True)
    single = np.flatnonzero(cnt[inv.ravel()] == 1)
    return single % len(F), edges[single]


SEAM_TOL = 0.5  # voxels: rim vertices of two blocks closer than this are the same vertex


def weld_seams(P: np.ndarray, F: np.ndarray, block: np.ndarray, tol: float
               ) -> tuple[np.ndarray | None, int]:
    """(point map, number of points moved) welding the cut rims of neighbouring blocks.

    Every GPU block writes the iso-surface of its own cells up to the centre plane of the
    neighbour's first cell layer, where the neighbour's surface starts. Both rims lie on
    the same grid edges of that plane but at different places along them (tak360 step 8,
    two blocks: median 0.07, 99 % < 0.34, max 0.8 voxels; the neighbour's rim is the
    linear interpolation of its own cell values to 1e-4 voxels, the other block's is
    not: it interpolates other values there, its halo copy). This is no round-off, so
    a rim vertex (on an open edge) is joined with the nearest rim vertex of another
    block within tol; joined groups (2 vertices, up to 6 where the rims cross different
    grid edges) collapse to their first point (map[i] = the point that replaces i).
    Only rim vertices at a seam move (observed: median 0.07, max 1.3 voxels); vertices
    inside a block and rims at walls are never touched, so the smallest feature affected
    is the cut rim of a structure crossing a seam. Needs scipy: (None, -1) without it."""
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree
    except ImportError:
        return None, -1
    tri, ends = open_edges(F, len(P))
    if not len(tri):
        return None, 0
    rim_block = np.repeat(block[tri], 2)
    _, at = np.unique(ends.ravel() * (int(block.max()) + 1) + rim_block, return_index=True)
    rim, rim_block = ends.ravel()[at], rim_block[at]  # one entry per (point, block)
    a, b = [], []
    for k in np.unique(rim_block):
        mine, others = rim[rim_block == k], rim[rim_block != k]
        if not len(mine) or not len(others):
            continue
        d, j = cKDTree(P[others]).query(P[mine], distance_upper_bound=tol)
        j = np.minimum(j, len(others) - 1)
        hit = np.isfinite(d) & (others[j] != mine)
        a.append(mine[hit])
        b.append(others[j[hit]])
    if not sum(len(x) for x in a):
        return None, 0
    a, b = np.concatenate(a), np.concatenate(b)
    used, inv = np.unique(np.r_[a, b], return_inverse=True)
    inv = inv.ravel()
    _, lab = connected_components(coo_matrix((np.ones(len(a), np.int8), (inv[:len(a)], inv[len(a):])),
                                             shape=(len(used), len(used))), directed=False)
    to = np.arange(len(P))
    to[used] = used[np.unique(lab, return_index=True)[1]][lab]  # used is sorted: first = lowest index
    return to, int((to != np.arange(len(P))).sum())


def surface_regions(pv, P: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, F, region id per triangle) of the connected pieces of a triangle surface,
    numbered like pyvista's connectivity('all') (VTK): by cell count, largest first, among
    equal counts the piece whose first triangle comes later first. With scipy the pieces
    come from the point graph (connected_components), else from the VTK filter."""
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        mesh = pv.PolyData(P, np.c_[np.full(len(F), 3), F].ravel()).connectivity(extraction_mode="all")
        _, rid = np.unique(np.asarray(mesh.cell_data["RegionId"]), return_inverse=True)
        return np.asarray(mesh.points, float), mesh.faces.reshape(-1, 4)[:, 1:], rid.ravel()
    n = len(F)
    graph = coo_matrix((np.ones(2 * n, np.int8), (np.r_[F[:, 0], F[:, 1]], np.r_[F[:, 1], F[:, 2]])),
                       shape=(len(P), len(P)))
    n_comp, lab = connected_components(graph, directed=False)
    comp = lab[F[:, 0]]
    cnt = np.bincount(comp, minlength=n_comp)
    first = np.full(n_comp, n)
    first[comp[::-1]] = np.arange(n)[::-1]  # first triangle of each piece
    used = np.flatnonzero(cnt)
    new = np.zeros(n_comp, np.intp)
    new[used[np.lexsort((-first[used], -cnt[used]))]] = np.arange(len(used))
    return P, F, new[comp]


def analyze_vof_surface(pv, surf, sampler=None, voxel_size: float | None = None,
                        n_samples: int = 25) -> dict:
    """Per-structure volume, area, closedness and inside fluid of a VOF iso-surface.

    Degenerate triangles (M-Star pads its output with them) are dropped and
    duplicate points merged (within 1e-3 voxels). With a 'block' cell array (the
    blocks of a multi-GPU output, set by vof_structures) and a voxel size, the cut
    rims of neighbouring blocks are welded (weld_seams, SEAM_TOL voxels), so that
    structures crossing a block seam can be closed. Structures come from an M-Star
    structure/region ID array if present, else from connectivity. Volume uses the
    divergence theorem; a structure is closed if no edge belongs to only one triangle.

    "inside" is the fluid-1 volume fraction inside each closed structure
    (NaN for open ones): sampler(points) at up to n_samples points just inside
    its surface ("inside_surface"), replaced for drops < 4 voxels across by the
    sample at the volume-weighted centroid ("inside_centroid"), where the
    surface-offset points can land in interface cells. Without a sampler it is
    inferred from the triangle orientation (M-Star orients the iso-surface
    normals toward fluid 2).
    Returns {"vol", "area", "closed", "inside", "inside_surface",
    "inside_centroid", "signed", "source", "n_welded"} (n_welded: points moved by the
    seam weld, -1 if it could not run without scipy).
    """
    empty = np.zeros(0)
    none = {"vol": empty, "area": empty, "closed": empty.astype(bool),
            "inside": empty, "inside_surface": empty, "inside_centroid": empty,
            "signed": empty, "source": "none", "n_welded": 0}
    if surf.n_cells == 0:
        return none
    if not surf.is_all_triangles:
        surf = surf.triangulate()
    f = surf.faces.reshape(-1, 4)[:, 1:]
    raw = np.asarray(surf.points, float)
    first, merge = merge_points(raw, 1e-3 * voxel_size if voxel_size else None)
    P = raw[first]
    f = merge[f]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    if not keep.any():
        return none
    F, kept_cells, n_welded = f[keep], np.flatnonzero(keep), 0
    block = np.asarray(surf.cell_data["block"]).ravel()[keep] if "block" in surf.cell_data.keys() else None
    if voxel_size and block is not None and block.min() != block.max():
        to, n_welded = weld_seams(P, F, block, SEAM_TOL * voxel_size)
        if n_welded > 0:
            F = to[F]
            keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
            F, kept_cells = F[keep], kept_cells[keep]

    id_name = None
    for data, per_point in ((surf.cell_data, False), (surf.point_data, True)):
        for name in data.keys():
            if "structure" in name.lower() or "region" in name.lower():
                id_name = name
                vals = np.asarray(data[name])
                if per_point:
                    vals = vals[surf.faces.reshape(-1, 4)[:, 1]]
                sid = vals[kept_cells]
                break
        if id_name:
            break

    if id_name:
        _, rid = np.unique(np.asarray(sid).ravel(), return_inverse=True)
        rid = rid.ravel()
        source = f"structure ID array '{id_name}'"
    else:
        P, F, rid = surface_regions(pv, P, F)
        source = "connectivity"
    n = int(rid.max()) + 1

    p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    tet = np.einsum("ij,ij->i", p0, np.cross(p1, p2)) / 6.0
    signed = np.bincount(rid, tet, n)
    vol = np.abs(signed)
    nrm = np.cross(p1 - p0, p2 - p0)
    area = np.bincount(rid, 0.5 * np.linalg.norm(nrm, axis=1), n)

    open_struct = np.zeros(n, bool)
    open_struct[rid[open_edges(F, len(P))[0]]] = True
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
            "signed": signed, "source": source, "n_welded": n_welded}


PHASE_METRICS = [  # (per-fluid time-series CSV column, summary variable template)
    ("n_drops", "Number of Drops n_drops ({} drops) [count]"),
    ("d32_m", "Sauter Diameter d32 (VOF, {} drops) [m]"),
    ("d32_dx", "Sauter Diameter d32 / Voxel Size (VOF, {} drops) [m/m]"),
    ("d10_m", "Mean Drop Diameter d10 (VOF, {} drops) [m]"),
    ("d43_m", "Volume-Mean Drop Diameter d43 (VOF, {} drops) [m]"),
    ("d_max_m", "Largest Drop Diameter (VOF, {} drops) [m]"),
    ("dmax_d32", "Largest / Sauter Drop Diameter d_max/d32 (VOF, {} drops) [m/m]"),
    ("F_disp", "Dispersed Fraction F_disp ({} in drops / {} volume) [m^3/m^3]"),
]


def volume_vti(name: str) -> bool:
    """A .vti of the Volume output (lower-case file name), not the BoundaryConditions snapshot."""
    return not name.startswith("boundaryconditions")


def run_frames(func, jobs: list[tuple], workers: int, what: str) -> list:
    """[func(*job) for job in jobs], with workers > 1 in that many processes (spawned:
    no VTK or BLAS state is inherited). The results keep the order of the jobs, so every
    number equals the sequential run. job[0] is the list of the frame's files; a frame
    that raises, or a worker that dies, ends the analysis with an error naming the frame."""
    workers = max(1, min(int(workers or 1), len(jobs)))
    if workers == 1:
        return [func(*job) for job in jobs]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    log(f"  {what}: {len(jobs)} frames in {workers} worker processes")
    sys.stdout.flush()
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        futures = [pool.submit(func, *job) for job in jobs]
        out = []
        for job, fut in zip(jobs, futures):
            try:
                out.append(fut.result())
            except BrokenProcessPool as e:
                raise RuntimeError(f"{what}: a worker process died (killed? out of memory?) at or "
                                   f"before frame {job[0][0].name}; use fewer --workers") from e
            except Exception as e:
                raise RuntimeError(f"{what}: frame {job[0][0].name} failed in a worker: "
                                   f"{type(e).__name__}: {e}") from e
        return out
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def vof_frame(paths: list[Path], t_name: float | None, vti_frames: list[tuple], idx1: int,
              voxel_size: float, v_min: float, phases: list[tuple],
              a_int: tuple[np.ndarray, np.ndarray] | None) -> dict:
    """One VOF surface frame of vof_structures (all blocks of one output time): its
    vof_structures.csv record "rec", its drops "dsd" [(phase label, region id, d_eq,
    volume, area)] and what vof_structures reports about it; {"skip": message} without
    a frame time. Runs in a worker process with --workers."""
    pv = import_pyvista("VOF structure analysis")
    meshes = [pv.read(p) for p in paths]
    t, src = frame_time(meshes[0], t_name)
    if t is None:
        return {"skip": f"  {paths[0].name}: no time in field data or filename; skipped"}
    if len(meshes) > 1:
        for i, m in enumerate(meshes):
            m.cell_data["block"] = np.full(m.n_cells, i, np.int32)
    surf = meshes[0] if len(meshes) == 1 else pv.merge(meshes)
    # the Volume output of the same time holding the fluid's fraction array
    sampler = next((s for tv, p in vti_frames if abs(tv - t) <= 1e-6 * max(1.0, abs(t))
                    for s in [vti_sampler(pv, p, idx1)] if s is not None), None)
    s = analyze_vof_surface(pv, surf, sampler, voxel_size)
    vol, area, closed = s["vol"], s["area"], s["closed"]

    debris = closed & (vol < v_min)
    kept = closed & ~debris
    is1 = s["inside"] > 0.5
    checks = [0, 0, 0, 0]  # drops checked, .vti vs orientation disagree, with centroid, centroid vs surface disagree
    if sampler:
        both = kept & np.isfinite(s["inside_centroid"])
        checks = [int(kept.sum()), int((kept & (is1 != (s["signed"] > 0))).sum()), int(both.sum()),
                  int((both & ((s["inside_centroid"] > 0.5) != (s["inside_surface"] > 0.5))).sum())]
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
    dsd = []
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
                        f"d_max_m_{tag}": float(d.max()),
                        f"dmax_d32_{tag}": float(d.max()) / d32})
        else:
            rec.update({f"{c}_{tag}": np.nan for c in ("d32_m", "d32_dx", "d10_m",
                                                        "d43_m", "d_max_m", "dmax_d32")})
        dsd.extend((name, int(gid), di, vi, ai) for gid, di, vi, ai in zip(np.flatnonzero(sel), d, V, A))
    return {"rec": rec, "dsd": dsd, "src": src, "id_source": s["source"], "checks": checks,
            "phase_source": f"fluid-{idx1} volume fraction in .vti" if sampler
            else "triangle orientation (no matching .vti)",
            "n_blocks": len(meshes), "n_welded": s["n_welded"]}


def vof_structures(case_dir: Path, min_voxels: float, voxel_size: float,
                   t_start: float, fluids: list[tuple[int, str, np.ndarray, np.ndarray]],
                   a_int: tuple[np.ndarray, np.ndarray] | None = None,
                   out_dir: Path | None = None,
                   min_d_dx: float = 4.0, workers: int = 1) -> tuple[list[dict], dict]:
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
    The frames are independent: workers > 1 analyses that many at a time (vof_frame).
    Returns (summary rows, plot series).
    """
    pv = import_pyvista("VOF structure analysis")
    if pv is None:
        return [], {}
    # VolumeVOFStructures/vof_structures.*.vtp is a point cloud, not the surface
    frames = complete_frames(find_frames(case_dir, ".vtp", lambda n: "vof" in n and "structure" not in n),
                             "VOF surface")
    if not frames:
        warn(f"--dispersion: no VOF surface .vtp files (name containing 'VOF', "
             f"excluding Slice outputs) found under {case_dir}; skipping VOF structures")
        return [], {}
    log(f"VOF structures: {len(frames)} frame(s) under {case_dir}")
    d_min = max((6.0 * min_voxels / np.pi) ** (1.0 / 3.0), min_d_dx) * voxel_size
    v_min = np.pi / 6.0 * d_min ** 3
    log(f"  drop filter: d_eq >= {d_min:.3g} m (stricter of {min_voxels:g} voxels "
        f"and {min_d_dx:g} dx); smaller closed structures are excluded")
    vti_frames = [(tv, p) for tv, p in complete_frames(find_frames(case_dir, ".vti", volume_vti),
                                                       "volume output") if tv is not None]
    idx1 = fluids[0][0]
    # label by fluid index so batch columns line up across fluid systems
    phases = [(f"fluid{i}", f"Fluid {i}", ft, fv) for i, _, ft, fv in fluids]
    log("  phases: " + ", ".join(f"Fluid {i} = {name}" for i, name, *_ in fluids))

    ts_rows, dsd_rows, dsd_plot = [], [], []
    sources, id_sources, phase_sources = set(), set(), set()
    n_blocks, welds, checks = 1, [], np.zeros(4, int)
    for fr in run_frames(vof_frame, [(paths, t_name, vti_frames, idx1, voxel_size, v_min, phases, a_int)
                                     for t_name, paths in frames], workers, "VOF structures"):
        if "skip" in fr:
            warn(fr["skip"])
            continue
        t = fr["rec"]["time_s"]
        sources.add(fr["src"])
        id_sources.add(fr["id_source"])
        phase_sources.add(fr["phase_source"])
        n_blocks = max(n_blocks, fr["n_blocks"])
        welds += [fr["n_welded"]] * (fr["n_blocks"] > 1)
        checks += fr["checks"]
        for name, gid, di, vi, ai in fr["dsd"]:
            dsd_rows.append({"time_s": t, "phase": name, "region_id": gid,
                             "d_eq_m": di, "volume_m3": vi, "area_m2": ai})
            dsd_plot.append((t, di, vi, name))
        ts_rows.append(fr["rec"])
    n_checked, n_disagree, n_cen, n_cen_dis = (int(c) for c in checks)
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
    if n_blocks > 1 and min(welds) < 0:
        warn(f"  VOF surface written in up to {n_blocks} blocks per frame and scipy is not "
             "installed: the blocks' cut rims on a seam are not welded, structures crossing "
             "a block seam are open and excluded, biasing d32 toward small drops")
    elif n_blocks > 1:
        log(f"  VOF surface written in up to {n_blocks} blocks per frame: the blocks' cut rims on a "
            f"seam differ by a fraction of a voxel, rim vertices within {SEAM_TOL:g} voxels were "
            f"welded ({np.mean(welds):.0f} per frame on average) so that structures crossing a "
            "block seam are whole; a rim without a partner leaves its structure open (excluded)")

    out_dir = out_dir or case_dir
    cols = ["time_s", "n_regions", "n_open", "n_debris", "n_resolved",
            "A_surface_m2", "A_drops_m2"] \
        + (["A_int_m2"] if a_int is not None else []) + ["coverage"] \
        + [f"{c}_{tag}" for tag, *_ in phases
           for c in ("n_drops", "V_drops_m3", "F_disp", "d32_m", "d32_dx",
                     "d10_m", "d43_m", "d_max_m", "dmax_d32")]
    write_csv(ts_rows, out_dir / "vof_structures.csv", columns=cols, what="frames", labels=True)
    write_csv(dsd_rows, out_dir / "vof_dsd.csv",
              columns=["time_s", "phase", "region_id", "d_eq_m", "volume_m3", "area_m2"],
              what="structures", labels=True)

    t = np.array([r["time_s"] for r in ts_rows])
    series = {"t": t, "dsd": dsd_plot, "phases": [(tag, name) for tag, name, *_ in phases]}
    for c in cols[1:]:
        series[c] = np.array([r.get(c, np.nan) for r in ts_rows], float)
    rows = []
    metrics = [(f"{key}_{tag}", var.format(name, name))
               for tag, name, *_ in phases for key, var in PHASE_METRICS]
    if a_int is not None:
        metrics.append(("coverage", "Drop Interface Coverage A_drops/A_int [m^2/m^2]"))
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
    rows.extend(dsd_shape(dsd_plot, [name for _, name, *_ in phases], t_start))
    return rows, series


def dsd_shape(dsd: list[tuple], phases: list[str], t_start: float) -> list[dict]:
    """Volume-weighted mean and standard deviation of X = d/d32 per fluid over the
    drops of all frames at t >= t_start (all frames if none), to compare with the
    normal volume distribution of HIM eq. (12-29): mean 1.07, std 0.23-0.24."""
    rows = []
    for name in phases:
        sel = [(d, v) for t, d, v, nm in dsd if nm == name and t >= t_start] \
            or [(d, v) for t, d, v, nm in dsd if nm == name]
        if len(sel) < 5:
            continue
        d, v = np.array([x[0] for x in sel]), np.array([x[1] for x in sel])
        x = d / ((d ** 3).sum() / (d ** 2).sum())
        m = float((v * x).sum() / v.sum())
        sd = float(np.sqrt((v * (x - m) ** 2).sum() / v.sum()))
        for var, val in ((f"DSD Volume Mean of d/d32 ({name} drops) [m/m]", m),
                         (f"DSD Volume Std of d/d32 ({name} drops) [m/m]", sd)):
            rows.append(value_row("vof_dsd.csv", var, val, "vof pooled", t_start, "", len(sel)))
        log(f"  DSD ({name}): volume mean d/d32 = {m:.3g}, std = {sd:.3g} "
            "(HIM eq. 12-29: 1.07, 0.24)")
    return rows


def region_shares(ndimage, blocks: list[dict]) -> list[float] | None:
    """Per fluid: voxels in the largest connected region of phi_k > 0.5 / all
    phi_k > 0.5 voxels, with the blocks (uniformity_frame) merged onto one grid (None
    if the blocks have different spacings)."""
    spacing = blocks[0]["spacing"]
    if any(not np.allclose(b["spacing"], spacing) for b in blocks):
        return None
    org = np.min([b["org"] for b in blocks], axis=0)
    offs = [np.rint((b["org"] - org) / spacing).astype(int) for b in blocks]
    gdims = np.max([o + b["dims"] for o, b in zip(offs, blocks)], axis=0)
    shares = []
    for k in range(len(blocks[0]["phi"])):
        grid = np.zeros(tuple(gdims[::-1]), bool)
        for o, b in zip(offs, blocks):
            dims = b["dims"]
            grid[o[2]:o[2] + dims[2], o[1]:o[1] + dims[1], o[0]:o[0] + dims[0]] \
                |= (b["phi"][k] > 0.5) & b["fluid"]
        lab, _ = ndimage.label(grid)
        sizes = np.bincount(lab.ravel())[1:]
        shares.append(float(sizes.max() / sizes.sum()) if sizes.size and sizes.sum()
                      else np.nan)
    return shares


def fraction_arrays(pv, frames: list[tuple]) -> tuple[dict | None, dict[str, list[str]]]:
    """The phase-fraction arrays of the .vti output, from the first file holding one:
    ({"phi_name", "mask_name", "frac_names", "phases": [(array, column suffix, label)],
    "fnames": {label: fluid name}, "continuity": scipy is there for it}, {}) or
    (None, {output kind: its array names})."""
    seen: dict[str, list[str]] = {}
    for _, paths in frames:
        for p in paths:
            reader = pv.get_reader(str(p))
            cell_names = list(reader.cell_array_names)
            names = cell_names + list(reader.point_array_names)
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
            phases: list[tuple[str, str, str]] = []
            fnames: dict[str, str] = {}
            continuity = False
            if len(frac_names) >= 2:
                for n in frac_names:
                    m = re.search(r"fluid\s*(\d+)\s*(?:\(([^)]*)\))?", n, re.I)
                    phases.append((n, f"_fluid{m.group(1)}", f"Fluid {m.group(1)}"))
                    fnames[f"Fluid {m.group(1)}"] = (m.group(2) or "").strip()
                try:
                    from scipy import ndimage  # noqa: F401
                    continuity = True
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
            log(f"  phase fraction: " + ", ".join(f"'{n}'" for n, *_ in phases)
                + f" from {p.parent.parent.name}"
                + (f", non-fluid mask: '{mask_name}' (nonzero = solid)" if mask_name
                   else ", non-fluid mask: sum of fluid volume fractions < 0.5"
                   if frac_names else ""))
            return {"phi_name": phases[0][0], "mask_name": mask_name, "frac_names": frac_names,
                    "phases": phases, "fnames": fnames, "continuity": continuity}, {}
    return None, seen


def fluid_indices(fluid: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """For the True voxels of fluid[z, y, x], in C order: the index along array axis
    `axis` and the flat index over the two other axes."""
    idx = np.nonzero(fluid)
    a, b = [i for i in range(3) if i != axis]
    small = np.int32 if fluid.size < 2 ** 31 else np.intp
    return idx[axis].astype(small), (idx[a] * fluid.shape[b] + idx[b]).astype(small)


def uniformity_frame(paths: list[Path], t_name: float | None, arrays: dict,
                     scales: list[tuple[int, int]], ax: int, axis_center: list[float] | None,
                     continuity_threshold: float) -> dict | None:
    """One .vti frame of phase_uniformity (all blocks of one output time): its
    phase_uniformity.csv record "rec", holdup_profile.csv rows "prof" and
    phase_continuity.csv record "cont" (None: not computed; False: blocks with
    different spacings); "skip" = True for a frame without fluid voxels; None if no
    block holds the fraction array. Runs in a worker process with --workers.

    The statistics are sums over the fluid voxels per zone (axial x radial) and per
    layer. A voxel's zone follows from its layer index and its column index (position
    in the two horizontal axes) through small per-layer / per-column tables, so no
    per-voxel coordinates are built; the sums run over the voxels in file order
    (block by block), like one pass over the concatenated blocks."""
    pv = import_pyvista("phase uniformity analysis")
    phi_name, mask_name, frac_names, phases = (arrays[k] for k in ("phi_name", "mask_name",
                                                                   "frac_names", "phases"))
    h1, h2 = [a for a in range(3) if a != ax]
    blocks, bounds = [], []
    t = src = None
    for p in paths:
        reader = pv.get_reader(str(p))
        cell_names = list(reader.cell_array_names)
        names = cell_names + list(reader.point_array_names)
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
        phi = [np.asarray(data[n]).reshape(shape) for n, *_ in phases]
        fluid = np.isfinite(phi[0])
        for a in phi[1:]:
            fluid &= np.isfinite(a)
        if mask_name is not None and mask_name in data.keys():
            fluid &= np.asarray(data[mask_name]).reshape(shape) == 0
        elif frac_names and all(n in data.keys() for n in frac_names):
            # M-Star writes no solid flag; solid cells have all fluid fractions = 0
            total = np.asarray(data[frac_names[0]], float) if len(frac_names) == 1 \
                else np.add(data[frac_names[0]], data[frac_names[1]], dtype=float)
            for n in frac_names[2:]:
                total += np.asarray(data[n])
            fluid &= total.reshape(shape) > 0.5
            del total
        c1d = [mesh.origin[a] + (np.arange(dims[a]) + off) * mesh.spacing[a]
               for a in range(3)]
        blocks.append({"phi": phi, "fluid": fluid, "c1d": c1d, "dz": mesh.spacing[ax],
                       "org": np.array(mesh.origin) + off * np.array(mesh.spacing),
                       "spacing": np.array(mesh.spacing), "dims": dims, "mesh": mesh})
    if not blocks or t is None:
        return None

    b = np.array(bounds)
    if axis_center:
        c1, c2 = axis_center
    else:
        c1 = (b[:, 2 * h1].min() + b[:, 2 * h1 + 1].max()) / 2
        c2 = (b[:, 2 * h2].min() + b[:, 2 * h2 + 1].max()) / 2
    va = 2 - ax  # array axis of the vertical coordinate
    v_lo, v_hi, r1 = [], [], []
    for blk in blocks:
        zz, yy, xx = np.meshgrid(blk["c1d"][2], blk["c1d"][1], blk["c1d"][0], indexing="ij", sparse=True)
        coord = (xx, yy, zz)
        blk["rr"] = np.sqrt((coord[h1] - c1) ** 2 + (coord[h2] - c2) ** 2)  # per column
        lay = blk["c1d"][ax][blk["fluid"].any(axis=tuple(i for i in range(3) if i != va))]
        if lay.size:
            v_lo.append(lay.min())
            v_hi.append(lay.max())
            r1.append(blk["rr"][blk["fluid"].any(axis=va, keepdims=True)].max())
    out = {"t": t, "src": src, "skip": not v_lo, "cont": None}
    if not v_lo:
        return out
    v_lo, v_hi, r1 = min(v_lo), max(v_hi), max(r1)
    for blk in blocks:
        blk["lay"], blk["col"] = fluid_indices(blk["fluid"], va)

    def over_voxels(per_block):
        parts = [per_block(blk) for blk in blocks]
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    ph = [over_voxels(lambda blk, k=k: blk["phi"][k][blk["fluid"]].astype(float))
          for k in range(len(phases))]
    dz = blocks[0]["dz"]
    z0, z1 = v_lo - dz / 2, v_hi + dz / 2
    two_fluid = bool(phases[0][2])
    rec = {"time_s": t}
    for (_, sfx, _), p in zip(phases, ph):
        rec[f"phi_mean{sfx}"] = float(p.mean())
    for nzz, nrr in scales:
        ssfx = f"_{nzz}x{nrr}"
        zone = over_voxels(lambda blk: (
            np.minimum(((blk["c1d"][ax] - z0) / (z1 - z0) * nzz).astype(int), nzz - 1)[blk["lay"]] * nrr
            + np.minimum((blk["rr"] / max(r1, 1e-30) * nrr).astype(int), nrr - 1).ravel()[blk["col"]]))
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
        del zone
    out["rec"] = rec

    layer = over_voxels(lambda blk: np.rint((blk["c1d"][ax] - v_lo) / dz).astype(int)[blk["lay"]])
    lw = np.bincount(layer)
    prof = {k: {"time_s": t, "z_m": v_lo + k * dz} for k in np.flatnonzero(lw)}
    for (_, sfx, _), p in zip(phases, ph):
        lsum = np.bincount(layer, p)
        for k, r in prof.items():
            r[f"phi_mean{sfx}"] = lsum[k] / lw[k]
    out["prof"] = list(prof.values())
    del ph, layer, p
    for blk in blocks:
        del blk["lay"], blk["col"]

    if two_fluid and arrays["continuity"]:
        from scipy import ndimage
        shares = region_shares(ndimage, blocks)
        if shares is None:
            out["cont"] = False
        else:
            cont = [lab for (_, _, lab), s in zip(phases, shares)
                    if s > continuity_threshold]
            crec = {"time_s": t, "continuous_phase":
                    "both (stratified or co-continuous)" if len(cont) > 1
                    else f"{cont[0]} ({arrays['fnames'][cont[0]]})" if cont else "none"}
            crec.update({f"share{sfx}": s for (_, sfx, _), s in zip(phases, shares)})
            out["cont"] = crec
    return out


def phase_uniformity(case_dir: Path, n_zones_z: int, n_zones_r: int, t_start: float,
                     vertical_axis: str = "y", axis_center: list[float] | None = None,
                     out_dir: Path | None = None,
                     zone_scales: list[tuple[int, int]] | None = None,
                     window: float = 1.0, tol: float = 0.02,
                     continuity_threshold: float = 0.9, workers: int = 1) -> tuple[list[dict], dict]:
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
    The frames are independent: workers > 1 analyses that many at a time
    (uniformity_frame).
    """
    pv = import_pyvista("phase uniformity analysis")
    if pv is None:
        return [], {}
    frames = complete_frames(find_frames(case_dir, ".vti", volume_vti), "volume output")
    if not frames:
        warn(f"--uniformity: no .vti volume files found under {case_dir}; skipping")
        return [], {}
    scales = zone_scales or [(n_zones_z, n_zones_r)]
    ax = "xyz".index(vertical_axis)
    log(f"phase uniformity: {len(frames)} frame(s) under {case_dir}, "
        f"vertical axis {vertical_axis}, zone scales "
        + ", ".join(f"{a}x{b}" for a, b in scales))

    arrays, seen = fraction_arrays(pv, frames)
    phases: list[tuple[str, str, str]] = arrays["phases"] if arrays else []  # (array, column suffix, label)
    cov_rows, prof_rows, cont_rows, sources = [], [], [], set()
    spacings_differ = False
    for fr in run_frames(uniformity_frame, [(paths, t_name, arrays, scales, ax, axis_center, continuity_threshold)
                                            for t_name, paths in frames] if arrays else [],
                         workers, "phase uniformity"):
        if fr is None:
            continue
        sources.add(fr["src"])
        if fr["skip"]:
            warn(f"  t = {fr['t']:g} s: no fluid voxels; skipped")
            continue
        cov_rows.append(fr["rec"])
        prof_rows.extend(fr["prof"])
        if fr["cont"] is False and not spacings_differ:
            warn("  .vti blocks have different spacings; phase continuity skipped")
        spacings_differ |= fr["cont"] is False
        if fr["cont"] and not spacings_differ:
            cont_rows.append(fr["cont"])
    if arrays is None:
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
              + ([f"S{k}" for k, _ in combos] if two_fluid else []), what="frames",
              labels=True)
    write_csv(prof_rows, out_dir / "holdup_profile.csv",
              columns=["time_s", "z_m"] + [f"phi_mean{s}" for s in sfxs], what="rows",
              labels=True)

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
                         f"Zonal CoV of Phase Fraction ({label}) [m^3/m^3]",
                         t, series[f"cov{key}"], t_start, "zonal")
        if row:
            rows.append(row)
        if not two_fluid:
            continue
        s = np.array([r[f"S{key}"] for r in cov_rows], float)
        series[f"S{key}"] = s
        row = series_row("phase_uniformity.csv", f"Segregation Index S ({label}) [m^3/m^3]",
                         t, s, t_start, "zonal")
        if row:
            rows.append(row)
        kin = approach_kinetics(t, s, window_f, tol, levels=(0.95,), fallback=True)
        if kin and np.isfinite(kin["t95"]):
            rows.append(value_row("phase_uniformity.csv",
                                  f"Homogeneity Time t_S ({label}) [s]", kin["t95"],
                                  "S(t) kinetics" if kin["plateau"]
                                  else "S(t) kinetics (no plateau, last window)",
                                  kin["t_ref"], float(t[-1]), int(t.size)))
            if not kin["plateau"]:
                warn(f"  S(t) ({label}) never plateaus; homogeneity time from the "
                     f"last-window mean of S (t >= {kin['t_ref']:.4g} s), provisional")
        elif kin:
            warn(f"  S(t) ({label}) never plateaus; no homogeneity time")

    if cont_rows:
        ct = np.array([r["time_s"] for r in cont_rows])
        write_csv(cont_rows, out_dir / "phase_continuity.csv",
                  columns=["time_s"] + [f"share{s}" for s in sfxs] + ["continuous_phase"],
                  what="frames", labels=True)
        series["share"] = {}
        for _, sfx, label in phases:
            series["share"][label] = np.array([r[f"share{sfx}"] for r in cont_rows])
            row = series_row("phase_continuity.csv", f"Largest Region Share ({label}) [m^3/m^3]",
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


def eps_series(files: list[Path], eps_file: str | None = None, what: str = "Hinze d_max",
               flag: str = "--dispersion") -> tuple[str, str, np.ndarray, np.ndarray] | None:
    """(source name, column, t, eps) of the energy dissipation rate from
    Fluid.txt or eps_file, preferring the instantaneous W/kg column; None
    (with a warning naming `what`) if unavailable."""
    if eps_file:
        p = Path(eps_file)
        src = [p] if p.is_file() else [f for f in files if eps_file.lower() in str(f).lower()]
        if not src:
            warn(f"--eps-file '{eps_file}' matched no stats file; {what} skipped")
            return None
    else:
        src = [f for f in files if f.name.lower() == "fluid.txt"]
    table = load_stats_table(src[0]) if src else None
    if table is None:
        warn(f"{flag}: no {'--eps-file' if eps_file else 'Fluid.txt'} stats "
             f"table for the {what}; skipped")
        return None
    header, data = table
    src_name = f"{src[0].parent.name}/{src[0].name}" if eps_file else src[0].name
    cands = [j for j in range(1, len(header)) if "energy dissipation" in norm_name(header[j])]
    if not cands:
        warn(f"{flag}: no energy dissipation column in {src_name}; {what} skipped")
        return None
    j = min(cands, key=lambda j: (var_unit(header[j]).lower() != "w/kg",
                                  "time-avg" in header[j].lower()))
    return src_name, header[j].strip(), data[:, 0], data[:, j]


def hinze_dmax(eps, sigma: float, rho_c: float, c_hinze: float = 0.725,
               k_phi: float = 0.0, phi: float = 0.0, eps_ratio: float = 1.0):
    """Hinze maximum stable drop size; NaN where eps <= 0."""
    eps = np.asarray(eps, float) * eps_ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        d = c_hinze * (1 + k_phi * phi) * (sigma / rho_c) ** 0.6 \
            * np.where(eps > 0, eps, np.nan) ** -0.4
    return float(d) if d.ndim == 0 else d


def dispersion_timeseries(itf: dict, out_path: Path, files: list[Path],
                          voxel_size: float | None, sigma: float | None = None,
                          rho_c: float | None = None, c_hinze: float = 0.725,
                          k_phi: float = 0.0, eps_ratio: float = 1.0,
                          eps_file: str | None = None, phi: float | None = None
                          ) -> list[dict]:
    """Per-time-point dispersion metrics from interface_metrics() series: A_int,
    V_liq, a, enhancement E = a/a0, a_norm = (a - a0)/(a_ss - a0), cumulative
    integral of a dt, d32 (interface) and, with sigma and rho_c, the Hinze
    d_max (and d_max/dx) from the instantaneous eps(t). Writes out_path."""
    if itf.get("a") is None:
        return []
    t, a = np.asarray(itf["t"], float), np.asarray(itf["a"], float)
    kin = itf.get("kinetics") or {}
    a0, a_ss = kin.get("a0", np.nan), kin.get("a_ss", np.nan)
    nan = np.full_like(a, np.nan)
    ok = np.isfinite(t) & np.isfinite(a)
    int_a = nan.copy()
    if ok.any():
        tt, aa = t[ok], a[ok]
        cum = np.r_[0.0, np.cumsum(0.5 * (aa[1:] + aa[:-1]) * np.diff(tt))]
        # a = a0 (flat interface) before the first sample, as in dispersion_kinetics
        int_a[ok] = cum + (tt[0] * a0 if tt[0] > 0 and np.isfinite(a0) else 0.0)
    cols = {"time_s": t, "A_int_m2": itf["area"], "V_liq_m3": itf.get("vol", nan),
            "a_1_m": a,
            "E": a / a0 if np.isfinite(a0) and a0 > 0 else nan,
            "a_norm": (a - a0) / (a_ss - a0) if np.isfinite(a_ss) and a_ss != a0 else nan,
            "int_a_s_m": int_a}
    if "d32" in itf:
        cols["d32_interface_m"] = itf["d32"]
    if sigma is not None and rho_c is not None:
        es = eps_series(files, eps_file)
        if es is not None:
            _, col, te, ve = es
            fin = np.isfinite(te) & np.isfinite(ve)
            if fin.any():
                eps = np.interp(t, te[fin], ve[fin])
                d = hinze_dmax(eps, sigma, rho_c, c_hinze, k_phi,
                               phi if phi is not None else 0.0, eps_ratio)
                cols.update(eps_W_kg=eps, d_hinze_m=d)
                if voxel_size:
                    cols["d_hinze_dx"] = d / voxel_size
    rows = [dict(zip(cols, map(float, vals))) for vals in zip(*cols.values())]
    write_csv(rows, out_path, columns=list(cols), what="time points", labels=True)
    return rows


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
    es = eps_series(files, eps_file, "resolution check")
    if es is None:
        return []
    src_name, col, te, ve = es
    row = series_row(src_name, col, te, ve, t_start, "hinze")
    if row is None or row["mean"] <= 0:
        warn("--dispersion: no positive steady energy dissipation; resolution check skipped")
        return []
    eps = row["mean"]
    if k_phi and phi is None:
        warn("--k-phi given but the dispersed fraction is unknown; holdup correction skipped")
    phi_eff = phi if phi is not None else 0.0
    d_hinze = hinze_dmax(eps, sigma, rho_c, c_hinze, k_phi, phi_eff, eps_ratio)
    ratio = d_hinze / voxel_size
    log(f"resolution check: eps = {eps:.4g} from '{col}' in {src_name}, "
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
            {**common, "variable": "Hinze Drop Size / Voxel Size [m/m]", "unit": "m/m",
             "mean": ratio}]


SH_NEWMAN = 2 * np.pi ** 2 / 3     # rigid drop, long contact (Newman 1931)
SH_KRONIG_BRINK = 17.9            # circulating drop, long contact (Kronig & Brink 1950)


def mass_transfer(rows: list[dict], s: dict, kin: dict, phi: float | None,
                  diff_c: float | None, diff_d: float | None, m: float | None,
                  dc0: float | None, t_start: float, coverage: float | None = None) -> list[dict]:
    """Two-film estimate of liquid-liquid mass transfer from GLOBAL steady averages
    (one mean dissipation, the plateau a_ss and d32 = 6 phi/a_ss; not per cell):
    k_c (Lamont & Scott eddy-cell model; Skelland & Moeti, HIM eq. 12-66, per impeller
    as a cross-check), k_d for rigid (Newman) and circulating (Kronig & Brink) drops,
    1/K_oc = 1/k_c + 1/(m k_d), K_oc a, the batch approach rate
    k = K_oc a [1/(1-phi) + 1/(m phi)], its t90 = ln 10 / k and, with dc0, the initial
    rate N0 = K_oc A_ss dc0. m = C_d*/C_c* (equilibrium distribution coefficient)."""
    if diff_c is None and diff_d is None:
        return []
    pos = lambda x: isinstance(x, (int, float)) and np.isfinite(x) and x > 0  # noqa: E731
    a_ss, a_tot = kin.get("a_ss", np.nan), kin.get("A_ss", np.nan)
    if not pos(a_ss) or not pos(phi) or phi >= 1:
        warn("mass transfer: needs the plateau a_ss and the dispersed fraction; skipped")
        return []
    if coverage is not None and coverage < 0.5:
        warn(f"mass transfer: closed drops carry only {100 * coverage:.2g}% of the interface; "
             "k_d from d32 = 6 phi/a_ss is not a drop property here")
    cont = re.match(r"Fluid (\d+)", str(s.get("Continuous phase (largest initial volume)", "")))
    ic = int(cont.group(1)) if cont else 1
    idp = 2 if ic == 1 else 1
    nu_c = s.get(f"Fluid {ic} kinematic viscosity [m^2/s]")
    rho_c, rho_d = s.get(f"Fluid {ic} density [kg/m^3]"), s.get(f"Fluid {idp} density [kg/m^3]")
    sigma, tank = s.get("Surface tension [N/m]"), s.get("Tank diameter T (domain width) [m]")
    by_var = {r["variable"]: r["mean"] for r in rows}
    eps, eps_src = by_var.get("Power-based Mean Dissipation P/M [W/kg]"), "power-based P/M"
    if not pos(eps):
        eps = next((r["mean"] for r in rows if str(r["file"]).lower() == "fluid.txt"
                    and norm_name(r["variable"]) == "mean energy dissipation rate"), None)
        eps_src = "resolved volume mean (Fluid.txt; excludes subgrid dissipation)"
    method = "two-film, global averages"
    out: list[dict] = []

    def add(var, val, note=method):
        if pos(val):
            out.append(value_row("mass transfer", var, float(val), note, t_start))
            log(f"  {var}: {val:.4g}")

    d32 = 6 * phi / a_ss
    add("Drop Diameter for Mass Transfer d32 = 6 phi/a_ss [m]", d32)
    k_c = None
    if pos(diff_c) and pos(nu_c):
        sc = nu_c / diff_c
        add("Continuous-Phase Schmidt Number Sc_c [(m^2/s)/(m^2/s)]", sc)
        if pos(eps):
            add("Mass Transfer Dissipation [W/kg]", eps, f"{method}: {eps_src}")
            k_c = 0.4 * (eps * nu_c) ** 0.25 * sc ** -0.5
            add("Continuous-Phase k_c Lamont-Scott [m/s]", k_c, f"{method}: {eps_src}")
        else:
            warn("mass transfer: no mean dissipation (run the base Stats summary); k_c skipped")
        if pos(rho_c) and pos(rho_d) and pos(sigma) and pos(tank):
            for key in [k for k in s if k.startswith("Impeller speed (")]:
                name = key[len("Impeller speed ("):key.rindex(")")]
                rpm, d_imp = speed_mag(s.get(key)), s.get(f"Impeller diameter ({name}) [m]")
                if not (pos(rpm) and pos(d_imp)):
                    continue
                n = rpm / 60.0
                re_i, fr = n * d_imp ** 2 / nu_c, n ** 2 * d_imp / 9.81
                sh = 1.237e-5 * sc ** (1 / 3) * re_i ** (2 / 3) * fr ** (5 / 12) \
                    * (d_imp / d32) ** 2 * (d32 / tank) ** 0.5 \
                    * (rho_d * d32 ** 2 * 9.81 / sigma) ** (5 / 4) * phi ** -0.5
                add(f"Continuous-Phase k_c Skelland-Moeti HIM eq. 12-66 ({name}) [m/s]", sh * diff_c / d32,
                    "correlation (impeller N, D, T and d32)")
    elif diff_c is not None:
        warn("mass transfer: continuous-phase viscosity unknown; k_c skipped")
    k_d = {}
    if pos(diff_d):
        k_d = {"rigid drops": SH_NEWMAN * diff_d / d32, "circulating drops": SH_KRONIG_BRINK * diff_d / d32}
        add("Dispersed-Phase k_d Rigid Drops (Newman) [m/s]", k_d["rigid drops"])
        add("Dispersed-Phase k_d Circulating Drops (Kronig-Brink) [m/s]", k_d["circulating drops"])
    if not (k_c and k_d):
        return out
    if not pos(m):
        warn("mass transfer: --partition (m = C_d*/C_c*) not given; K_oc a and extraction time skipped")
        return out
    for limit, kd in k_d.items():
        k_oc = 1.0 / (1.0 / k_c + 1.0 / (m * kd))
        add(f"Overall K_oc a, {limit} [1/s]", k_oc * a_ss)
        k = k_oc * a_ss * (1.0 / (1.0 - phi) + 1.0 / (m * phi))
        add(f"Extraction Rate Constant k, {limit} [1/s]", k)
        add(f"Extraction Time t90 = ln10/k, {limit} [s]", np.log(10.0) / k)
        add(f"Continuous-Film Resistance Share, {limit} [(s/m)/(s/m)]", (1.0 / k_c) / (1.0 / k_oc))
        if pos(dc0) and pos(a_tot):
            add(f"Initial Transfer Rate N0 = K_oc A_ss dC0, {limit} [mol/s]", k_oc * a_tot * dc0)
    return out


CELLS_RESOLVED = 10.0  # cells per d32 for a converged interface area (interface-capturing rule of thumb)
VOF_FILES = ("vof_structures.csv", "vof_dsd.csv")
UNI_FILES = ("phase_uniformity.csv", "phase_continuity.csv", "holdup_profile.csv")


def reuse_results(run_dirs: list[Path], case_id: str | None, files: tuple[str, ...],
                  out_base: Path | None) -> list[dict]:
    """Summary rows whose 'file' is one of files, from the first earlier output root
    holding them; the matching per-case CSVs are copied to out_base. case_id = the case
    of a --batch root (long tables batch_summary*.csv with a 'case' column, else the
    case's own <root>/<case_id>/stats_summary*.csv); None = the root of a single-case
    run (<root>/stats_summary*.csv, per-case CSVs in <root>/). A row keeps the
    '(reused: <run>)' tag of the run that computed it. [] if no run has them."""
    for d in run_dirs:
        src = Path(d) / case_id if case_id else Path(d)
        tables = (sorted(Path(d).glob("batch_summary*.csv")) if case_id else []) \
            + sorted(src.glob("stats_summary*.csv"))
        found: list[dict] = []
        for f in tables:
            if f.stem.endswith(("_means", "_settings")):
                continue
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("case", case_id) == case_id and r.get("file") in files:
                        found.append(r)
        if not found:
            continue
        rows, seen = [], set()
        for r in found:
            if r["variable"] in seen:
                continue
            seen.add(r["variable"])
            r = {k: r.get(k, "") for k in COLUMNS}
            try:
                r["mean"] = float(r["mean"])
            except ValueError:
                pass
            if "(reused: " not in r["steady_method"]:
                r["steady_method"] = f"{r['steady_method']} (reused: {Path(d).name})"
            rows.append(r)
        if out_base is not None:
            out_base.mkdir(parents=True, exist_ok=True)
            for name in files:
                if (src / name).is_file() and (src / name).resolve() != (out_base / name).resolve():
                    shutil.copyfile(src / name, out_base / name)
        log(f"  reused {len(rows)} rows ({', '.join(files)}) from {d}")
        return rows
    warn(f"  --reuse-results: no earlier {', '.join(files)} results"
         + (f" for {case_id}" if case_id else ""))
    return []
CELLS_MARGINAL = 4.0   # below this a drop is a grid-scale blob (also the default --min-d-dx)
S_UNIFORM, S_DISPERSED, S_PARTIAL = 0.1, 0.5, 0.8  # segregation-index classes
G = 9.81


def settling_velocity(d: float, drho: float, rho_c: float, mu_c: float) -> float:
    """Terminal velocity [m/s] of a rigid sphere (Schiller-Naumann drag, Newton's
    C_D = 0.44 above Re = 1000); fluid drops with internal circulation settle up to
    1.5x faster in the Stokes limit (Hadamard-Rybczynski), so this is a lower bound."""
    v = abs(drho) * G * d ** 2 / (18 * mu_c)
    for _ in range(100):
        re = max(rho_c * v * d / mu_c, 1e-12)
        cd = 24 / re * (1 + 0.15 * re ** 0.687) if re < 1000 else 0.44
        v_new = np.sqrt(4 * G * d * abs(drho) / (3 * rho_c * cd))
        if abs(v_new - v) <= 1e-9 * v_new:
            return float(v_new)
        v = 0.5 * (v + v_new)
    return float(v)


def dispersion_assessment(rows: list[dict], s: dict, kin: dict, phi: float | None,
                          dx: float | None, t_start: float, min_d_dx: float = CELLS_MARGINAL,
                          sigma: float | None = None, out_csv: Path | None = None,
                          diff_c: float | None = None, diff_d: float | None = None,
                          m: float | None = None) -> list[dict]:
    """How well the grid resolves the drops and how far the CFD interfacial area can be
    trusted: correlation drop sizes (HIM eqs. 12-30 and 12-28 combined, per impeller; the
    smallest governs), cells per drop, inertial-subrange check d32/eta, area bounds
    (correlation a = 6 phi/d32, the largest a the grid can hold, the CFD plateau a_ss),
    refinement needed for 10 cells per drop, settling vs mean velocity of correlation-size
    and CFD-size drops, approach to the equilibrium d32 (HIM eq. 12-33) and the dispersion
    state from the segregation index. Two corrections of the correlation area for incomplete
    dispersion: Estimate A (dispersed share f_d from S, a_eff = f_d a_ref) and Estimate B
    (grid-normalised CFD area a_ss dx, for ratios between cases); the k_c scale
    (eps nu_c)^(1/4) and, with diff_c, diff_d and m, K_oc at the reference d32 and K_oc a_eff.
    Writes a readable table of the checks to out_csv."""
    pos = lambda x: isinstance(x, (int, float)) and np.isfinite(x) and x > 0  # noqa: E731
    cont = re.match(r"Fluid (\d+)", str(s.get("Continuous phase (largest initial volume)", "")))
    ic = int(cont.group(1)) if cont else 1
    idp = 2 if ic == 1 else 1
    rho_c, rho_d = s.get(f"Fluid {ic} density [kg/m^3]"), s.get(f"Fluid {idp} density [kg/m^3]")
    mu_c, mu_d = s.get(f"Fluid {ic} dynamic viscosity [Pa s]"), s.get(f"Fluid {idp} dynamic viscosity [Pa s]")
    nu_c = s.get(f"Fluid {ic} kinematic viscosity [m^2/s]")
    sigma = sigma if pos(sigma) else s.get("Surface tension [N/m]")
    phi = phi if pos(phi) else s.get("Dispersed fraction phi (initial) [m^3/m^3]")
    v_liq = s.get("Fluid volume (total, initial) [m^3]")
    t_end = s.get("Simulated time (end) [s]")
    if not (pos(rho_c) and pos(sigma) and pos(phi)):
        warn("dispersion assessment: needs both fluids' properties, the surface tension and "
             "the dispersed fraction (input.xml); skipped")
        return []
    method = "dispersion assessment"
    out: list[dict] = []
    checks: list[dict] = []  # readable table

    def add(var, val, note=method, zero=False):
        if isinstance(val, str) or pos(val) or (zero and val == 0):
            out.append(value_row("dispersion assessment", var, val if isinstance(val, str) else float(val),
                                 note, t_start))
            log(f"  {var}: {val if isinstance(val, str) else f'{val:.4g}'}")

    def check(item, val, unit, verdict, basis):
        checks.append({"check": item, "value": val, "unit": unit, "assessment": verdict, "basis": basis})

    by_var = {r["variable"]: r["mean"] for r in rows}
    fluid_mean = lambda key: next((r["mean"] for r in rows if str(r["file"]).lower() == "fluid.txt"  # noqa: E731
                                   and norm_name(r["variable"]) == key), None)
    eps_avg = by_var.get("Power-based Mean Dissipation P/M [W/kg]")
    u_mean = fluid_mean("mean velocity")

    # correlation drop sizes per impeller; the smallest d32 (strongest impeller) governs
    imps = []
    for key in [k for k in s if k.startswith("Impeller speed (")]:
        name = key[len("Impeller speed ("):key.rindex(")")]
        rpm, d_imp = speed_mag(s.get(key)), s.get(f"Impeller diameter ({name}) [m]")
        if not (pos(rpm) and pos(d_imp)):
            continue
        n = rpm / 60.0
        we = rho_c * n ** 2 * d_imp ** 3 / sigma
        d_inv = d_imp * D32_C1 * (1 + D32_C3 * phi) * we ** -0.6
        visc = 1.0
        if pos(rho_d) and pos(mu_d):
            vi = (rho_c / rho_d) ** 0.5 * mu_d * n * d_imp / sigma
            visc = (1 + 0.92 * vi ** 0.84) ** 0.6
        eps_max = by_var.get(f"Maximum Dissipation Estimate P/(rho_c D^3) ({name}) [W/kg]")
        re_i = n * d_imp ** 2 / nu_c if pos(nu_c) else None
        imps.append({"name": name, "n": n, "D": d_imp, "We": we, "d_inv": d_inv, "visc": visc,
                     "d_ref": d_inv * visc, "eps_max": eps_max, "Re": re_i})
        add(f"d32 Correlation HIM eq. 12-30 ({name}) [m]", d_inv)
        add(f"Viscous-Drop Factor (1 + 0.92 Vi^0.84)^0.6 HIM eq. 12-28 ({name}) [m/m]", visc)
        add(f"d32 Reference Estimate ({name}) [m]", d_inv * visc)
    if not imps:
        warn("dispersion assessment: no impeller speed/diameter in input.xml; skipped")
        return []
    g = min(imps, key=lambda i: i["d_ref"])
    d_ref = g["d_ref"]
    add("d32 Reference Estimate [m]", d_ref, f"{method}: HIM eq. 12-30 x viscous factor of eq. 12-28, impeller {g['name']}")
    check("d32 reference estimate (correlation)", d_ref, "m", f"impeller {g['name']}, We = {g['We']:.3g}",
          "HIM eq. (12-30) d32 = 0.053 D (1 + 3 phi) We^-0.6 x viscous factor (1 + 0.92 Vi^0.84)^0.6 of eq. (12-28)")

    # validity of the correlation
    notes = []
    if phi > 0.3:
        notes.append(f"phi = {phi:.2f} > 0.3 (outside the eq. 12-30 data; coalescence likely raises d32)")
    if g["Re"] is not None and g["Re"] < 1e4:
        notes.append(f"Re = {g['Re']:.3g} < 1e4 (not fully turbulent)")
    if g["visc"] > 1.1:
        notes.append(f"viscous factor {g['visc']:.2f} (eq. 12-28 is for dilute dispersions)")
    eta_max = (nu_c ** 3 / g["eps_max"]) ** 0.25 if pos(nu_c) and pos(g["eps_max"]) else None
    if eta_max:
        r_eta = d_ref / eta_max
        add("d32 Reference / Kolmogorov Scale at eps_max [m/m]", r_eta)
        check("d32 reference / Kolmogorov scale at eps_max", r_eta, "m/m",
              "inertial subrange (correlation applies)" if r_eta > 1 else
              "drops smaller than the Kolmogorov scale: viscous-subrange breakup, correlation uncertain",
              "Hinze/Weber scaling holds for eta << d (HIM p. 664)")
        if r_eta <= 1:
            notes.append("d32 < Kolmogorov scale")
    check("correlation validity", "; ".join(notes) or "within range", "",
          "use with care" if notes else "ok",
          "noncoalescing, phi < 0.3, turbulent, inertial subrange (HIM pp. 661-664)")
    if notes:
        warn("  correlation outside its range: " + "; ".join(notes))

    # approach to the equilibrium d32 within the simulated time
    if pos(t_end):
        nt = g["n"] * t_end
        add("Impeller Turns Simulated N t [count]", nt)
        factor = (3.8e3 / min(max(nt, 100.0), 3.8e3)) ** (1 / 6)
        add("Transient d32 Factor (3800/(N t))^(1/6) HIM eq. 12-33 [m/m]", factor)
        check("impeller turns simulated", nt, "count",
              "equilibrium drop size reached" if nt >= 3.8e3 else
              f"short of equilibrium: d32 about {factor:.2f} x the equilibrium value"
              + (" (N t < 100: before the correlation range)" if nt < 100 else ""),
              "HIM eq. (12-33): d32(t)/d32_inf = (3.8e3/(N t))^(1/6), 100 < N t < 3800")

    # grid resolution of the drops
    resolution = None
    if pos(dx):
        cells = d_ref / dx
        add("d32 Reference / Voxel Size [m/m]", cells)
        resolution = ("resolved" if cells >= CELLS_RESOLVED else "marginal" if cells >= CELLS_MARGINAL
                      else "under-resolved (grid-limited)" if cells >= 1 else "sub-grid (grid-limited)")
        add("Drop Resolution", resolution)
        check("cells per drop, d32 reference / dx", cells, "m/m", resolution,
              f">= {CELLS_RESOLVED:g} cells resolved, {CELLS_MARGINAL:g}-{CELLS_RESOLVED:g} marginal, "
              f"< {CELLS_MARGINAL:g} grid-scale blobs, < 1 not representable (interface-capturing rule of thumb)")
        if cells < CELLS_RESOLVED:
            refine = CELLS_RESOLVED * dx / d_ref
            add("Grid Refinement for 10 Cells per d32 [m/m]", refine)
            add("Lattice Cost Factor for 10 Cells per d32 [count]", refine ** 4)
            check("grid refinement for 10 cells per d32", refine, "m/m",
                  f"dx would have to be {d_ref / CELLS_RESOLVED:.3g} m; about {refine ** 4:.3g} x the "
                  "cell updates (cells ~ refine^3, time steps ~ refine)",
                  "dx_req = d32_ref / 10")

    # CFD drops vs the correlation
    lab = f"Fluid {idp}"
    d_cfd = by_var.get(f"Sauter Diameter d32 (VOF, {lab} drops) [m]")
    if pos(d_cfd):
        r_cfd = d_cfd / d_ref
        add("CFD d32 / d32 Reference [m/m]", r_cfd)
        d_dx = d_cfd / dx if pos(dx) else None
        grid_set = d_dx is not None and d_dx < 2 * CELLS_RESOLVED and r_cfd > 2
        check(f"CFD d32 ({lab} drops) / d32 reference", r_cfd, "m/m",
              f"CFD drops are set by the grid ({d_dx:.3g} cells): the correlation d32 is the better estimate"
              if grid_set else "CFD and correlation agree within a factor of 2" if 0.5 <= r_cfd <= 2
              else "CFD drops differ from the correlation (check coalescence and the dissipation)",
              "resolved drops cluster a few cells above the debris filter when the physical drops are sub-grid")

    # interfacial area bounds
    a_ss = kin.get("a_ss", np.nan)
    a_ref = 6 * phi / d_ref
    add("Specific Area at d32 Reference a = 6 phi/d32 [1/m]", a_ref)
    if pos(v_liq):
        add("Interfacial Area at d32 Reference A = a V_liq [m^2]", a_ref * v_liq)
    if pos(dx):
        a_grid = 6 * phi / (max(min_d_dx, 1.0) * dx)
        add("Max Specific Area Representable on Grid [1/m]", a_grid)
        check("largest specific area the grid can hold, 6 phi/(min_d_dx dx)", a_grid, "1/m",
              "below the correlation a: the CFD area is capped by the grid" if a_grid < a_ref
              else "the grid can hold the correlation area",
              f"all of phi as drops of the smallest counted size ({min_d_dx:g} dx)")
    if pos(a_ss):
        cap = a_ss / a_ref
        add("CFD / Correlation Specific Area a_ss/a_ref [(1/m)/(1/m)]", cap)
        check("CFD plateau a_ss / correlation a", cap, "(1/m)/(1/m)",
              "CFD captures most of the expected area" if cap >= 0.5 else
              f"CFD resolves {100 * cap:.2g} % of the area of a full dispersion at the correlation d32",
              "a_ref = 6 phi / d32_ref (HIM eq. 12-2), fully dispersed, equilibrium")

    # settling of correlation-size and CFD-size drops vs the mean flow
    if pos(rho_d) and pos(mu_c) and rho_d != rho_c:
        drho = rho_d - rho_c
        for label, d in (("d32 Reference", d_ref), ("CFD d32", d_cfd)):
            if not pos(d):
                continue
            vt = settling_velocity(d, drho, rho_c, mu_c)
            add(f"Settling Velocity, {label} Drops [m/s]", vt)
            if pos(u_mean):
                add(f"Settling / Mean Velocity, {label} Drops [(m/s)/(m/s)]", vt / u_mean)
                check(f"settling / mean velocity, {label.lower()} drops", vt / u_mean, "(m/s)/(m/s)",
                      "drops follow the flow" if vt / u_mean < 0.05 else
                      "drops settle against the mean flow (segregation)" if vt / u_mean > 0.2 else
                      "weak segregation by settling",
                      f"rigid-sphere terminal velocity (Schiller-Naumann), {'rising' if drho < 0 else 'settling'}; "
                      "mean velocity from Fluid.txt")

    # dispersion state from the segregation index and the phase continuity
    s_idx = next((r["mean"] for r in rows if str(r["variable"]).startswith("Segregation Index S (")), None)
    shares = [r["mean"] for r in rows if str(r["variable"]).startswith("Largest Region Share (")]
    f_disp = by_var.get(f"Dispersed Fraction F_disp ({lab} in drops / {lab} volume) [m^3/m^3]")
    state = None
    if pos(s_idx) or s_idx == 0:
        state = ("uniform" if s_idx < S_UNIFORM else "dispersed, non-uniform" if s_idx < S_DISPERSED
                 else "partially dispersed" if s_idx < S_PARTIAL else "stratified (free layer)")
        add("Dispersion State", state, f"{method}: segregation index S = {s_idx:.3g}")
        check("dispersion state (segregation index S)", s_idx, "m^3/m^3", state,
              f"S < {S_UNIFORM} uniform, < {S_DISPERSED} dispersed but graded, < {S_PARTIAL} partially "
              "dispersed, otherwise a free layer remains (Danckwerts intensity of segregation)")
    else:
        check("dispersion state", "", "", "unknown: run with phase uniformity (--uniformity)", "")
    if shares:
        check("largest connected region share, both phases", min(shares), "m^3/m^3",
              "both phases connected (stratified, co-continuous, or drops below the grid)"
              if min(shares) > 0.9 else "one phase broken up", "phase_continuity.csv")
    if pos(f_disp):
        check(f"{lab} volume in resolved drops", f_disp, "m^3/m^3",
              "most of the dispersed phase is in resolved drops" if f_disp > 0.5
              else "little of it in resolved drops: sub-grid drops or a layer", "vof_structures")

    # Estimate A: the vessel as a free layer of the dispersed phase (share x of the liquid)
    # plus a uniform dispersion; S^2 = x (1 - phi) / ((1 - x) phi) inverted for x
    a_eff = f_d = None
    if pos(s_idx) or s_idx == 0:
        r_s = min(s_idx, 1.0) ** 2 * phi / (1 - phi)
        x_layer = r_s / (1 + r_s)
        f_d = min(max(1 - x_layer / phi, 0.0), 1.0)
        a_eff = f_d * a_ref
        add("Free-Layer Share x, Estimate A [m^3/m^3]", x_layer, f"{method}: S = {s_idx:.3g}, phi = {phi:.3g}")
        add("Dispersed Share f_d = 1 - x/phi, Estimate A [m^3/m^3]", f_d, zero=True)
        add("Effective Specific Area a_eff = f_d a_ref, Estimate A [1/m]", a_eff)
        check("Estimate A: dispersed share f_d from S (free layer + uniform dispersion)", f_d, "m^3/m^3",
              f"a_eff = f_d a_ref = {a_eff:.4g} 1/m; a graded (not layered) dispersion reads as partly "
              "layered, so f_d is low for well-mixed cases; a layer thinner than one zone is averaged away",
              "S^2 = x (1 - phi)/((1 - x) phi), r = S^2 phi/(1 - phi), x = r/(1 + r), f_d = 1 - x/phi [App]")
    # Estimate B: unresolved structures are always a few cells across (c ~ CFD d32/dx), so
    # a_ss dx ~ 6 phi_b / c measures how much liquid is torn apart; compare cases as ratios
    if pos(a_ss) and pos(dx):
        add("Grid-Normalised CFD Area a_ss dx, Estimate B [m/m]", a_ss * dx,
            f"{method}: extent of break-up; ratio between cases x fineness a_ref ratio = area ratio")
        a0 = kin.get("a0", np.nan)
        if pos(a0) and a_ss > a0:
            add("Grid-Normalised CFD Area (a_ss - a0) dx, Estimate B [m/m]", (a_ss - a0) * dx)
        c_cells = d_cfd / dx if pos(d_cfd) else None
        check("Estimate B: grid-normalised CFD area a_ss dx (extent)", a_ss * dx, "m/m",
              "compare between cases only: area ratio = (a_ss dx) ratio x a_ref ratio (fineness)"
              + (f"; CFD structures are {c_cells:.2g} cells across here" if c_cells else ""),
              "a_ss ~ 6 phi_b/(c dx) with c (cells per structure) the same in every case [App]")

    # transfer-coefficient scales: k_c ~ (eps nu_c)^(1/4) Sc^-1/2 (Lamont-Scott), k_d ~ D_d/d32
    if pos(eps_avg):
        add("Mean Dissipation for k_c, power-based P/M [W/kg]", eps_avg)
        if pos(nu_c):
            add("Lamont-Scott Velocity Scale (eps nu_c)^(1/4) [m/s]", (eps_avg * nu_c) ** 0.25,
                f"{method}: k_c = 0.4 (eps nu_c)^(1/4) Sc^-1/2; ratio between cases = k_c ratio at equal Sc")
    if pos(diff_c) and pos(diff_d) and pos(m) and pos(eps_avg) and pos(nu_c):
        k_c = 0.4 * (eps_avg * nu_c) ** 0.25 * (nu_c / diff_c) ** -0.5
        for limit, sh in (("rigid drops", SH_NEWMAN), ("circulating drops", SH_KRONIG_BRINK)):
            k_d = sh * diff_d / d_ref
            k_oc = 1.0 / (1.0 / k_c + 1.0 / (m * k_d))
            add(f"Overall K_oc at d32 Reference, {limit} [m/s]", k_oc,
                f"{method}: 1/K_oc = 1/k_c + 1/(m k_d), k_d = Sh D_d/d32_ref")
            add(f"Continuous-Film Resistance Share at d32 Reference, {limit} [(s/m)/(s/m)]", k_oc / k_c)
            if pos(a_eff):
                add(f"Overall K_oc a_eff, Estimate A, {limit} [1/s]", k_oc * a_eff)

    # verdict on the interfacial area
    lo = hi = best = None
    sub = resolution is not None and resolution != "resolved" and resolution != "marginal"
    if pos(a_ss):
        lo, hi = min(a_ss, a_ref), max(a_ss, a_ref)
        if state == "stratified (free layer)":
            best, verdict = a_ss, ("not dispersed: the CFD area (layer interface + entrained drops) "
                                   "applies; the correlation area does not")
        elif state in ("uniform", "dispersed, non-uniform") and sub:
            best, verdict = a_ref, ("dispersed but drops are below the grid: the CFD area is a lower "
                                    "bound; use the correlation area")
        elif resolution in ("resolved", "marginal") and state in ("uniform", "dispersed, non-uniform"):
            best, verdict = a_ss, "drops resolved: the CFD area is usable"
        elif state == "partially dispersed" and pos(a_eff):
            best, verdict = a_eff, ("partially dispersed: Estimate A, a_eff = f_d a_ref with f_d = "
                                    f"{f_d:.2g} from S (free layer + uniform dispersion)")
        else:
            verdict = ("partially dispersed or state unknown: the area lies between the CFD "
                       "value and the correlation value")
        add("Specific Area Lower Bound [1/m]", lo)
        add("Specific Area Upper Bound [1/m]", hi)
        if best is not None:
            add("Best-Estimate Specific Area [1/m]", best)
        add("Interfacial Area Assessment", verdict)
        check("interfacial area", best if best is not None else f"{lo:.4g} to {hi:.4g}", "1/m", verdict,
              "bounds = CFD plateau a_ss and correlation a_ref")
    if out_csv is not None and checks:
        write_csv(checks, out_csv, columns=["check", "value", "unit", "assessment", "basis"], what="checks")
    return out


# ---------------------------------------------------------------- gas-liquid mass transfer (free surface)

# Constants of the kL models. The original papers are not openly accessible; each value was
# checked against an open source that prints the equation with its citation (named below),
# except KL_THOMAS, which is NOT checked: it is the value of the solver-side variable hL_JT
# of the kLa cases.
KL_EDDY_CELL = 0.4                      # Lamont & Scott (1970): kL = 0.4 (eps nu)^(1/4) Sc^-1/2 [Maltby et al. 2018, eq. 30]
KL_PENETRATION = 2.0 / np.sqrt(np.pi)   # Higbie (1935): kL = 2 (D / (pi t_c))^(1/2); 1.13 [Maltby et al. 2018, eqs. 2, 29]
KL_KAWASE = 0.301                       # Kawase, Halard & Moo-Young (1992) [Maltby et al. 2018; Correia 2007]; M-Star's kLa guide: 0.3
KL_RIGID = 0.13                         # Calderbank & Moo-Young (1961): kL Sc^(2/3) = 0.13 [(P/V) mu/rho^2]^(1/4) [Correia 2007]
# kL = c (D beta_rms)^(1/2): c = 0.5 (Turney, Smith & Banerjee 2005); 0.525, 0.59, 0.60 in
# other flows [all four as compiled by Wissink & Herlina 2016, p. 535]
KL_SURFACE_DIVERGENCE = 0.5
KL_THOMAS = 0.19                        # Sh = 0.19 Re^0.8 Sc^(1/3), Re = u_s D_imp / nu (Thomas et al. 2023): not checked
# kL = c (eps nu)^(1/4) Sc^n: (name, c, n). "Kastanek" = Higbie with the Kolmogorov time
# (nu/eps)^(1/2) as contact time, the model of Kastanek (1977) and Kawase et al. (1987)
# [Nedeltchev & Schumpe 2011]; the rigid-surface form (Sc^-2/3: immobile interface) is the
# bound for a surface immobilised by surfactants
KL_EDDY_MODELS = [("Lamont-Scott", KL_EDDY_CELL, -0.5), ("Kawase-Moo-Young", KL_KAWASE, -0.5),
                  ("Kastanek", KL_PENETRATION, -0.5),
                  ("Calderbank-Moo-Young Rigid Surface", KL_RIGID, -2.0 / 3.0)]
DNS_DX_ETA = 2.1      # grid spacing of a DNS in Kolmogorov (or Batchelor) scales (Pope 2000: k_max eta >= 1.5)
SURFACE_DEPTH = 1.5   # voxels below the free surface at which the liquid fields are sampled
GL_FILE = "gas-liquid mass transfer"
GL_ASSESS = "gas-liquid assessment"
GL_SURFACE_CSV = "gas_liquid_surface.csv"
GL_FILES = (GL_SURFACE_CSV,)
GL_BULK, GL_STATS, GL_LOCAL = "bulk dissipation", "surface mean (Stats)", "local (VTK)"
GL_EPS_BULK = "Bulk Dissipation for kL [W/kg]"
GL_EPS_STATS = "Free-Surface Dissipation, area mean (Stats) [W/kg]"
GL_EPS_LOCAL = "Free-Surface Dissipation, effective for kL (VTK) [W/kg]"


def kl_eddy(eps, nu: float, diff: float, c: float = KL_EDDY_CELL, n: float = -0.5):
    """Small-eddy (eddy cell) liquid-side coefficient kL = c (eps nu)^(1/4) Sc^n [m/s];
    NaN where eps <= 0."""
    eps = np.asarray(eps, float)
    with np.errstate(invalid="ignore"):
        k = c * (np.where(eps > 0, eps, np.nan) * nu) ** 0.25 * (nu / diff) ** n
    return float(k) if k.ndim == 0 else k


def kl_higbie(u, length: float, diff: float):
    """Higbie penetration theory, kL = 2 (D / (pi t_c))^(1/2) [m/s], with the contact time
    t_c = length / u of liquid moving at speed u along the surface; NaN where u <= 0."""
    u = np.asarray(u, float)
    with np.errstate(invalid="ignore"):
        k = KL_PENETRATION * np.sqrt(diff * np.where(u > 0, u, np.nan) / length)
    return float(k) if k.ndim == 0 else k


def kl_thomas(u, d_imp: float, nu: float, diff: float):
    """Free-surface correlation Sh = 0.19 Re^0.8 Sc^(1/3) with Re = u_s D_imp / nu and
    kL = Sh D / D_imp [m/s]; u_s = mean speed of the free surface. NaN where u <= 0."""
    u = np.asarray(u, float)
    with np.errstate(invalid="ignore"):
        k = KL_THOMAS * diff / d_imp * (np.where(u > 0, u, np.nan) * d_imp / nu) ** 0.8 \
            * (nu / diff) ** (1.0 / 3.0)
    return float(k) if k.ndim == 0 else k


def kl_surface_divergence(beta_rms, diff: float):
    """Surface-divergence model kL = c (D beta_rms)^(1/2) [m/s]; beta = divergence of the
    velocity in the plane of the free surface [1/s]. NaN where beta_rms <= 0."""
    b = np.asarray(beta_rms, float)
    with np.errstate(invalid="ignore"):
        k = KL_SURFACE_DIVERGENCE * np.sqrt(diff * np.where(b > 0, b, np.nan))
    return float(k) if k.ndim == 0 else k


_VTK_DTYPES = {"Float32": "<f4", "Float64": "<f8", "Int8": "i1", "UInt8": "u1", "Int16": "<i2",
               "UInt16": "<u2", "Int32": "<i4", "UInt32": "<u4", "Int64": "<i8", "UInt64": "<u8"}


def _xml_attrs(tag: str) -> dict:
    return dict(re.findall(r'(\w+)="([^"]*)"', tag))


class VtiBlock:
    """Cell arrays of one .vti file (one block of the Volume output), read on demand.
    M-Star writes them raw, appended and uncompressed: such an array is memory-mapped, so
    sampling it at the free surface reads only the cells touched and not the whole grid.
    Any other encoding is read through pyvista (the whole array)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(path, "rb") as fh:
            head = fh.read(1 << 19)
        i = head.find(b"<AppendedData")
        txt = (head[:i] if i >= 0 else head).decode(errors="replace")
        root, img = re.search(r"<VTKFile[^>]*>", txt), re.search(r"<ImageData[^>]*>", txt)
        if not root or not img:
            raise ValueError(f"{self.path.name}: not an XML image data (.vti) file")
        root, img = _xml_attrs(root.group(0)), _xml_attrs(img.group(0))
        e = [int(v) for v in img["WholeExtent"].split()]
        self.spacing = np.array([float(v) for v in img.get("Spacing", "1 1 1").split()])
        self.origin = np.array([float(v) for v in img.get("Origin", "0 0 0").split()]) \
            + np.array(e[::2]) * self.spacing
        self.dims = np.array([e[1] - e[0], e[3] - e[2], e[5] - e[4]], np.int64)  # cells in x, y, z
        self._raw = i >= 0 and b'encoding="raw"' in head[i:i + 100] and "compressor" not in root \
            and root.get("byte_order", "LittleEndian") == "LittleEndian"
        self._start = head.index(b"_", i) + 1 if i >= 0 else 0
        self._hdr = 8 if root.get("header_type") == "UInt64" else 4
        self._size = self.path.stat().st_size
        sec = re.search(r"<CellData[^>]*>(.*?)</CellData>", txt, re.S)
        self.arrays: dict[str, tuple[str, int, str, int]] = {}  # name -> (type, components, format, offset)
        for tag in re.findall(r"<DataArray[^>]*>", sec.group(1) if sec else ""):
            a = _xml_attrs(tag)
            self.arrays[a.get("Name", "")] = (a.get("type", ""), int(a.get("NumberOfComponents", 1)),
                                              a.get("format", ""), int(a.get("offset", -1)))
        self._cache: dict[str, np.ndarray] = {}

    def array(self, name: str) -> np.ndarray:
        """(cells, components) values of a cell array in file order (x fastest, then y, z)."""
        if name not in self._cache:
            typ, ncomp, fmt, off = self.arrays[name]
            n = int(self.dims.prod())
            start = self._start + off + self._hdr
            if self._raw and fmt == "appended" and typ in _VTK_DTYPES and off >= 0 \
                    and start + n * ncomp * np.dtype(_VTK_DTYPES[typ]).itemsize <= self._size:
                arr = np.memmap(self.path, dtype=_VTK_DTYPES[typ], mode="r", offset=start, shape=(n, ncomp))
            else:
                import pyvista as pv
                reader = pv.get_reader(str(self.path))
                reader.disable_all_cell_arrays()
                reader.disable_all_point_arrays()
                reader.enable_cell_array(name)
                arr = np.asarray(reader.read().cell_data[name]).reshape(n, ncomp)
            self._cache[name] = arr
        return self._cache[name]


class CellSampler:
    """Nearest-cell lookup in the cell arrays of the Volume output (one VtiBlock per GPU
    block). A cell is addressed by (block index, flat cell index); block -1 = outside."""

    def __init__(self, blocks: list[VtiBlock]):
        self.blocks = blocks
        self.spacing = blocks[0].spacing

    def locate(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blk = np.full(len(pts), -1, np.int64)
        flat = np.zeros(len(pts), np.int64)
        for b, vb in enumerate(self.blocks):
            ijk = np.floor((pts - vb.origin) / vb.spacing).astype(np.int64)
            ok = (blk < 0) & np.all((ijk >= 0) & (ijk < vb.dims), axis=1)
            i, j, k = ijk[ok].T
            flat[ok] = (k * vb.dims[1] + j) * vb.dims[0] + i
            blk[ok] = b
        return blk, flat

    def centres(self, blk: np.ndarray, flat: np.ndarray) -> np.ndarray:
        out = np.full((len(blk), 3), np.nan)
        for b, vb in enumerate(self.blocks):
            sel = blk == b
            f, nx, ny = flat[sel], vb.dims[0], vb.dims[1]
            out[sel] = vb.origin + (np.stack([f % nx, (f // nx) % ny, f // (nx * ny)], axis=1) + 0.5) * vb.spacing
        return out

    def values(self, name: str, blk: np.ndarray, flat: np.ndarray) -> np.ndarray:
        """(points, components) values; NaN outside the grid and in blocks without the array."""
        ncomp = next(vb.arrays[name][1] for vb in self.blocks if name in vb.arrays)
        out = np.full((len(blk), ncomp), np.nan)
        for b, vb in enumerate(self.blocks):
            sel = np.flatnonzero(blk == b)
            if not sel.size or name not in vb.arrays:
                continue
            order = np.argsort(flat[sel], kind="stable")  # ascending file offsets
            out[sel[order]] = vb.array(name)[flat[sel][order]]
        return out


def volume_fields(blocks: list[VtiBlock]) -> dict[str, str | None]:
    """Names of the cell arrays the free-surface analysis samples: the instantaneous
    velocity vector, the energy dissipation rate (instantaneous preferred, else M-Star's
    time average) and the fill level of the free-surface model ('Volume Fraction')."""
    names: dict[str, int] = {}
    for vb in blocks:
        for n, (_, ncomp, _, _) in vb.arrays.items():
            names.setdefault(n, ncomp)
    inst = lambda n: "time-avg" not in n.lower()  # noqa: E731
    vel = [n for n, c in names.items() if c == 3 and n.lower().startswith("velocity") and inst(n)]
    eps = sorted((n for n, c in names.items() if c == 1 and "energy dissipation rate" in n.lower()),
                 key=lambda n: not inst(n))
    frac = [n for n, c in names.items() if c == 1 and re.match(r"volume fraction", n, re.I)]
    return {"velocity": vel[0] if vel else None, "eps": eps[0] if eps else None,
            "fraction": frac[0] if frac else None}


def below_surface(sampler: CellSampler, fields: dict, c: np.ndarray, n: np.ndarray,
                  w: np.ndarray, depth: float) -> dict:
    """Area-weighted means of the liquid fields at `depth` [m] below the free surface, for
    surface triangles with centroids c, unit normals n and areas w.

    The liquid side of the surface is the one where more of the area finds liquid: a cell
    is liquid when its velocity is non-zero (M-Star writes zeros in the gas and in solids)
    and, when the output holds the fill level, 'Volume Fraction' > 0.5. Triangles whose
    sample is not liquid (surface at a wall, thin films) are left out of the means.
    Returns {"sampled": area share with a liquid sample, "side", "u": mean speed,
    "u_sqrt": mean of speed^(1/2), "eps": mean dissipation, "eps_q": mean of eps^(1/4),
    "beta_rms": rms surface divergence, "sampled_beta": its area share}.

    Surface divergence beta = div u - n . grad u . n (the divergence of the velocity in the
    plane of the surface): grad u by central differences over the six neighbours of the
    sampled cell, used only where all six are liquid."""
    vel, eps_name, frac = fields["velocity"], fields["eps"], fields["fraction"]

    def liquid(blk, flat):
        u = sampler.values(vel, blk, flat)
        speed = np.linalg.norm(u, axis=1)
        with np.errstate(invalid="ignore"):
            liq = (blk >= 0) & (speed > 0)
            if frac:
                liq &= sampler.values(frac, blk, flat)[:, 0] > 0.5
        return u, speed, liq

    best = None
    for side in (-1.0, 1.0):
        blk, flat = sampler.locate(c + side * depth * n)
        u, speed, liq = liquid(blk, flat)
        if best is None or w[liq].sum() > best[0]:
            best = (float(w[liq].sum()), side, blk, flat, speed, liq)
    share, side, blk, flat, speed, liq = best
    out = {"sampled": share / float(w.sum()), "side": side}
    if not liq.any():
        return out
    wl, nl = w[liq], n[liq]
    out.update(u=float(np.average(speed[liq], weights=wl)),
               u_sqrt=float(np.average(np.sqrt(speed[liq]), weights=wl)))
    if eps_name:
        e = sampler.values(eps_name, blk, flat)[liq, 0]
        ok = np.isfinite(e) & (e >= 0)
        if ok.any():
            out.update(eps=float(np.average(e[ok], weights=wl[ok])),
                       eps_q=float(np.average(e[ok] ** 0.25, weights=wl[ok])))
    span = int(flat.max()) + 1
    cells, inv = np.unique(blk[liq] * span + flat[liq], return_inverse=True)
    cb, cf = cells // span, cells % span
    cen, h = sampler.centres(cb, cf), sampler.spacing
    grad = np.zeros((len(cells), 3, 3))  # grad[:, a, b] = d u_a / d x_b
    good = np.ones(len(cells), bool)
    for b in range(3):
        step = np.zeros(3)
        step[b] = h[b]
        up, _, liq_up = liquid(*sampler.locate(cen + step))
        dn, _, liq_dn = liquid(*sampler.locate(cen - step))
        good &= liq_up & liq_dn
        grad[:, :, b] = (up - dn) / (2.0 * h[b])
    g = grad[inv.ravel()]
    beta = np.trace(g, axis1=1, axis2=2) - np.einsum("ni,nij,nj->n", nl, g, nl)
    ok = good[inv.ravel()] & np.isfinite(beta)
    if ok.any():
        out.update(beta_rms=float(np.sqrt(np.average(beta[ok] ** 2, weights=wl[ok]))),
                   sampled_beta=float(wl[ok].sum() / w.sum()))
    return out


def surface_frame(paths: list[Path], t_name: float | None, vti_frames: list[tuple],
                  voxel: float) -> dict:
    """One VOF surface frame of free_surface_local (all blocks of one output time):
    {"time_s", "A": surface area, "A_closed", "n_closed": closed structures (bubbles in
    the liquid, droplets above it), "n_triangles", the surface means of the point data
    M-Star wrote on the surface ("pt_eps", "pt_eps_q", "pt_u", "pt_u_sqrt") and, with the
    Volume output of the same time, the below_surface() means}; {"skip": message} without
    a frame time. Holds no fluid properties: the kL models are applied by the caller.
    Runs in a worker process with --workers."""
    pv = import_pyvista("free-surface analysis")
    meshes = [pv.read(p) for p in paths]
    t, src = frame_time(meshes[0], t_name)
    if t is None:
        return {"skip": f"  {paths[0].name}: no time in field data or filename; skipped"}
    if len(meshes) > 1:
        for i, m in enumerate(meshes):
            m.cell_data["block"] = np.full(m.n_cells, i, np.int32)
    surf = meshes[0] if len(meshes) == 1 else pv.merge(meshes)
    if surf.n_cells and not surf.is_all_triangles:
        surf = surf.triangulate()
    st = analyze_vof_surface(pv, surf, None, voxel)
    rec = {"time_s": t, "src": src, "n_blocks": len(meshes), "A": float(st["area"].sum()),
           "A_closed": float(st["area"][st["closed"]].sum()), "n_closed": int(st["closed"].sum()),
           "n_triangles": 0}
    if not surf.n_cells:
        return rec
    F = surf.faces.reshape(-1, 4)[:, 1:]
    P = np.asarray(surf.points, float)
    p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    twice = np.linalg.norm(nrm, axis=1)
    ok = twice > 0  # M-Star pads the surface with degenerate triangles
    if not ok.any():
        return rec
    F, w, n, c = F[ok], 0.5 * twice[ok], nrm[ok] / twice[ok, None], (p0 + p1 + p2)[ok] / 3.0
    rec["n_triangles"] = int(ok.sum())
    for name in surf.point_data.keys():
        low, v = name.lower(), np.asarray(surf.point_data[name], float)
        if "time-avg" in low or len(v) != len(P):
            continue
        if "energy dissipation rate" in low and v.ndim == 1:
            e = np.maximum(v[F].mean(axis=1), 0.0)
            rec.update(pt_eps=float(np.average(e, weights=w)), pt_eps_q=float(np.average(e ** 0.25, weights=w)))
        elif low.startswith("velocity") and (v.ndim == 2 or "magnitude" in low):
            sp = (np.linalg.norm(v, axis=1) if v.ndim == 2 else np.abs(v))[F].mean(axis=1)
            rec.update(pt_u=float(np.average(sp, weights=w)), pt_u_sqrt=float(np.average(np.sqrt(sp), weights=w)))
    vti = next((p for tv, p in vti_frames if abs(tv - t) <= 1e-6 * max(1.0, abs(t))), None)
    if vti:
        try:
            blocks = [VtiBlock(p) for p in vti]
        except (ValueError, KeyError, OSError) as e:
            return {**rec, "note": f"  {vti[0].name}: unreadable ({e}); surface point data only"}
        fields = volume_fields(blocks)
        if fields["velocity"]:
            rec.update(below_surface(CellSampler(blocks), fields, c, n, w, SURFACE_DEPTH * voxel))
            rec["eps_array"] = fields["eps"]
    return rec


def free_surface_local(case_dir: Path, t_start: float, voxel: float, diff: float, nu: float,
                       vol: tuple[np.ndarray, np.ndarray], renewal: float | None,
                       d_imp: float | None, a_int: tuple[np.ndarray, np.ndarray] | None = None,
                       out_dir: Path | None = None, workers: int = 1) -> list[dict]:
    """Local estimate of the free-surface kLa from the VTK output (approach 2): for every
    Volume VOF Surface frame the kL models are evaluated with the liquid fields at the
    surface and integrated over it, kLa = (1/V_liq) sum_i kL_i dA_i:

      eddy models (KL_EDDY_MODELS)             c nu^(1/4) Sc^n <eps^(1/4)>
      Higbie                                   2 (D/(pi L))^(1/2) <u^(1/2)>   (t_c = L/u)
      Thomas                                   Sh = 0.19 Re^0.8 Sc^(1/3), Re = <u> D_imp/nu
      surface divergence                       0.5 (D beta_rms)^(1/2)

    <.> = area-weighted mean over the surface. The fields are those of the Volume .vti of
    the same time, SURFACE_DEPTH voxels below the surface (below_surface). When the Volume
    output is written less often than the VOF surface (fewer than half of the frames that
    hold a field have it), the point data M-Star wrote on the surface is used for every
    frame instead: one series never mixes the two positions. The surface divergence needs
    the Volume output. vol = (t, V_liq), a_int = (t, A) of the interface Stats file for
    comparison. Writes gas_liquid_surface.csv to out_dir and returns the summary rows
    (means over the frames at t >= t_start)."""
    if import_pyvista("free-surface analysis (local kL)") is None:
        return []
    frames = complete_frames(find_frames(case_dir, ".vtp", lambda n: "vof" in n and "structure" not in n),
                             "VOF surface")
    if not frames:
        warn(f"--gas-liquid: no VOF surface .vtp files (name containing 'VOF', excluding Slice "
             f"outputs) found under {case_dir}; local kL estimate skipped")
        return []
    log(f"free surface (local kL): {len(frames)} frame(s) under {case_dir}")
    vti_frames = [(tv, p) for tv, p in complete_frames(find_frames(case_dir, ".vti", volume_vti),
                                                       "volume output") if tv is not None]
    recs = []
    for fr in run_frames(surface_frame, [(paths, t_name, vti_frames, voxel) for t_name, paths in frames],
                         workers, "free surface"):
        if "skip" in fr:
            warn(fr["skip"])
            continue
        if "note" in fr:
            warn(fr["note"])
        recs.append(fr)
    recs = sorted((r for r in recs if r["A"] > 0), key=lambda r: r["time_s"])
    if not recs:
        warn("--gas-liquid: the VOF surface frames hold no surface; local kL estimate skipped")
        return []

    nan = float("nan")
    local = lambda name, unit: f"{name}, {GL_LOCAL} [{unit}]"  # noqa: E731
    steady = [r for r in recs if r["time_s"] >= t_start] or recs

    def source(key: str) -> str:
        """One source per field for all frames, so that the series is not a mix of two
        sampling positions: the Volume output when at least half of the frames that
        hold the field have it, else the point data on the surface ('' = neither)."""
        n_vol, n_pt = sum(key in r for r in steady), sum(f"pt_{key}" in r for r in steady)
        return "" if not (n_vol or n_pt) else "" if 2 * n_vol >= n_pt else "pt_"

    src_eps, src_u = source("eps_q"), source("u")
    has_eps = any(f"{src_eps}eps_q" in r for r in recs)
    has_u = any(f"{src_u}u" in r for r in recs)
    below = f"Volume .vti, {SURFACE_DEPTH:g} voxels below the surface"
    ts_rows = []
    for r in recs:
        t = r["time_s"]
        v_liq = float(np.interp(t, *vol))
        a = r["A"] / v_liq if v_liq > 0 else nan
        eps_q, eps_mean = r.get(f"{src_eps}eps_q", nan), r.get(f"{src_eps}eps", nan)
        u_mean, u_sqrt = r.get(f"{src_u}u", nan), r.get(f"{src_u}u_sqrt", nan)
        row = {"time [s]": t, "n_triangles [count]": r["n_triangles"], "A_surface [m^2]": r["A"],
               "A_closed [m^2]": r["A_closed"], "n_closed [count]": r["n_closed"],
               "V_liq [m^3]": v_liq, "a [1/m]": a,
               "sampled_area_share [m^2/m^2]": r.get("sampled", nan),
               "eps_surface_mean [W/kg]": eps_mean, "eps_surface_eff [W/kg]": eps_q ** 4,
               "u_surface [m/s]": u_mean, "beta_rms [1/s]": r.get("beta_rms", nan)}
        if a_int is not None:
            row["A_interface_stats [m^2]"] = float(np.interp(t, *a_int))
        kl = {name: kl_eddy(eps_q ** 4, nu, diff, c, n) for name, c, n in KL_EDDY_MODELS}
        if renewal:
            kl["Higbie"] = kl_higbie(u_sqrt ** 2, renewal, diff)
        if d_imp:
            kl["Thomas"] = kl_thomas(u_mean, d_imp, nu, diff)
        kl["Surface Divergence"] = kl_surface_divergence(r.get("beta_rms", nan), diff)
        for name, k in kl.items():
            row[local(f"kL {name}", "m/s")] = k
            row[local(f"kLa {name}", "1/s")] = k * a
        ts_rows.append(row)
    cols = list(ts_rows[0])
    for row in ts_rows:
        cols += [k for k in row if k not in cols]
    cols = [c for c in cols if any(np.isfinite(row.get(c, nan)) for row in ts_rows)]
    write_csv([{c: row.get(c, nan) for c in cols} for row in ts_rows],
              (out_dir or case_dir) / GL_SURFACE_CSV, columns=cols, what="frames")
    if has_eps:
        arrays = sorted({str(r["eps_array"]) for r in recs if r.get("eps_array")})
        log("  surface dissipation from: " + ("point data of the VOF surface" if src_eps
                                             else f"'{', '.join(arrays)}' of the {below}"))
    if has_u:
        log("  surface velocity from: " + ("point data of the VOF surface" if src_u else below))
    n_field = sum(f"{src_u}u" in r for r in steady)
    if has_u and n_field < len(steady):
        log(f"  {n_field} of {len(steady)} surface frame(s) at t >= {t_start:g} s have the fields; "
            "the others give the area only")
    if not has_eps:
        warn("--gas-liquid: neither the Volume output nor the VOF surface holds the 'Energy "
             "Dissipation Rate': the dissipation-based local kL (Lamont-Scott, Kastanek) is "
             "skipped; switch that variable on in the M-Star Volume output")
    if not has_u:
        warn("--gas-liquid: no Volume .vti at the times of the VOF surface frames and no velocity "
             "on the surface: only the surface area is taken from the VTK output")
    t = np.array([row["time [s]"] for row in ts_rows])
    n_after = int((t >= t_start).sum())
    if n_after < 10:
        warn(f"  only {n_after} VOF surface frame(s) at t >= {t_start:g} s; the local kL is "
             "poorly averaged (write more Volume VOF Surface output)")
    sampled = np.array([row["sampled_area_share [m^2/m^2]"] for row in ts_rows])
    if np.isfinite(sampled).any() and np.nanmean(sampled) < 0.8:
        warn(f"  only {100 * np.nanmean(sampled):.0f}% of the surface has liquid {SURFACE_DEPTH:g} "
             "voxels below it (thin films, surface at walls): the local means describe that part")
    series = lambda key: np.array([row.get(key, nan) for row in ts_rows], float)  # noqa: E731
    named = [("A_surface [m^2]", "Free-Surface Area (VOF surface) [m^2]"),
             ("n_closed [count]", "Closed Structures, Bubbles and Droplets (VOF surface) [count]"),
             ("eps_surface_mean [W/kg]", "Free-Surface Dissipation, area mean (VTK) [W/kg]"),
             ("eps_surface_eff [W/kg]", GL_EPS_LOCAL),
             ("u_surface [m/s]", "Free-Surface Speed (VTK) [m/s]"),
             ("beta_rms [1/s]", "Surface Divergence rms (VTK) [1/s]"),
             ("sampled_area_share [m^2/m^2]", "Sampled Share of the Free-Surface Area (VTK) [m^2/m^2]")]
    named += [(c, c) for c in cols if c.startswith(("kL ", "kLa "))]
    rows = []

    def method(key: str) -> str:
        """What a column was computed from: the surface alone, or its fields and their source."""
        eddy = key.startswith("eps") or any(f" {name}," in key for name, _, _ in KL_EDDY_MODELS)
        speed = key.startswith("u_") or " Higbie," in key or " Thomas," in key
        if eddy or speed:
            return "VOF surface + " + ("surface point data" if (src_eps if eddy else src_u) else below)
        return f"VOF surface + {below}" if key.startswith(("kL", "beta", "sampled")) else "VOF surface"

    for key, var in named:
        if key not in cols:
            continue
        row = series_row(GL_SURFACE_CSV, var, t, series(key), t_start, method(key))
        if row:
            rows.append(row)
            log(f"  {var}: {row['mean']:.5g} \u00b1 {row['std']:.3g}")
    with np.errstate(invalid="ignore", divide="ignore"):
        row = series_row(GL_SURFACE_CSV, "Closed-Structure Share of the Free-Surface Area (VOF surface) [m^2/m^2]",
                         t, series("A_closed [m^2]") / series("A_surface [m^2]"), t_start, "VOF surface")
    if row:
        rows.append(row)
    return rows


def bulk_dissipation(base: Path, files: list[Path], t: np.ndarray, vol: np.ndarray,
                     rho: float | None) -> tuple[np.ndarray | None, str]:
    """(mean dissipation of the liquid at times t [W/kg], its source): the power of all
    impellers over the liquid mass, P(t) / (rho V_liq(t)), which holds the resolved and
    the subgrid dissipation; without a torque record the resolved mean of Fluid.txt."""
    import xml.etree.ElementTree as ET
    p, imps = find_input_xml(base), []
    if p is not None:
        try:
            imps = impeller_info(ET.parse(p).getroot())
        except ET.ParseError:
            pass
    total = None
    for imp in imps:
        ps = power_series(files, imp)
        if ps is None:
            continue
        _, tp, pw = ps
        ok = np.isfinite(tp) & np.isfinite(pw)
        if ok.any():
            total = (0.0 if total is None else total) + np.interp(t, tp[ok], pw[ok])
    if total is not None and rho and rho > 0:
        eps = np.full_like(t, np.nan)
        np.divide(total, rho * vol, out=eps, where=vol > 0)
        return eps, "power-based P/M"
    es = eps_series(files, None, "bulk dissipation", "--gas-liquid")
    if es is not None:
        _, _, te, ve = es
        ok = np.isfinite(te) & np.isfinite(ve)
        if ok.any():
            return np.interp(t, te[ok], ve[ok]), "resolved volume mean (Fluid.txt; excludes subgrid dissipation)"
    return None, ""


def free_surface_stats(base: Path, files: list[Path], stats_dir: Path, s: dict, t_start: float,
                       diff: float, nu: float, renewal: float | None, d_imp: float | None,
                       v_liq: float | None = None, area_column: str | None = None,
                       out_path: Path | None = None) -> tuple[list[dict], dict]:
    """Free-surface kLa from the Stats files alone (approach 1), at every Stats sample:

      a(t) = A(t) / V_liq(t)            A = area of the free surface (interface Stats file)
      kL from the bulk dissipation      eps = P(t) / (rho V_liq): Lamont-Scott, Kawase-Moo-Young,
                                        Kastanek and the rigid-surface bound (KL_EDDY_MODELS)
      kL from the surface means that M-Star writes to the interface Stats file, when it
      holds them: 'EDR Mean' (the same models at the surface dissipation) and
      'Velocity Mean' (Higbie with t_c = renewal / u_s; Thomas, Sh = 0.19 Re^0.8 Sc^(1/3))

    Writes the time series to out_path and returns (summary rows = means over
    t >= t_start, {"t", "vol", "area"}); ([], {}) without a time base."""
    found = interface_table(files, stats_dir, area_column, "--gas-liquid")
    eps_s = u_s = area = None
    if found is not None:
        rel, header, data, j = found
        t, area = data[:, 0], data[:, j]
        normed = [norm_name(h) for h in header]
        je = next((k for k, h in enumerate(normed) if h in ("edr mean", "energy dissipation rate mean")), None)
        ju = next((k for k, h in enumerate(normed) if h == "velocity mean"), None)
        eps_s = data[:, je] if je is not None else None
        u_s = data[:, ju] if ju is not None else None
    else:
        fluid = [f for f in files if f.name.lower() == "fluid.txt"]
        table = load_stats_table(fluid[0]) if fluid else None
        if table is None:
            return [], {}
        rel, t = "Fluid.txt", table[1][:, 0]
    vol = np.full_like(t, v_liq) if v_liq is not None else fluid_volume_series(files, t)
    if vol is None:
        warn("--gas-liquid: no 'Fluid Volume' in Fluid.txt; give --liquid-volume")
        return [], {}
    cols: dict[str, np.ndarray] = {"time [s]": t}
    rows: list[dict] = []

    def add(file, var, values, method, key=None):
        cols[key or var] = values
        row = series_row(file, var, t, values, t_start, method)
        if row:
            rows.append(row)
            log(f"  {var}: {row['mean']:.5g} \u00b1 {row['std']:.3g}")

    a = None
    if area is not None:
        a = np.full_like(t, np.nan)
        np.divide(area, vol, out=a, where=vol > 0)
        add(rel, "Free-Surface Area A [m^2]", area, "interface", "A_surface [m^2]")
        cols["V_liq [m^3]"] = vol
        add(rel, "Specific Free-Surface Area a = A/V_liq [1/m]", a, "interface", "a [1/m]")
        fin = np.flatnonzero(np.isfinite(t) & np.isfinite(area))
        if fin.size and area[fin[0]] > 0:
            a0 = float(area[fin[0]])
            if fin.size > 1 and t[fin[0]] > np.median(np.diff(t[fin])):
                warn(f"--gas-liquid: first area sample at t = {t[fin[0]]:g} s, more than one output "
                     "interval after 0; A0 may not be the surface at rest")
            rows.append(value_row(rel, "Free-Surface Area at Rest A0 [m^2]", a0, "interface (first sample)",
                                  float(t[fin[0]]), float(t[fin[0]]), 1))
            add(rel, "Free-Surface Area Enhancement A/A0 [m^2/m^2]", area / a0, "interface", "A/A0 [m^2/m^2]")
    kls: dict[str, tuple[np.ndarray, str]] = {}  # model, basis -> (kL(t), what it was computed from)
    eps_b, eps_src = bulk_dissipation(base, files, t, vol, s.get("Fluid 1 density [kg/m^3]"))
    if eps_b is not None:
        add(GL_FILE, GL_EPS_BULK, eps_b, eps_src, "eps_bulk [W/kg]")
        for name, c, n in KL_EDDY_MODELS:
            kls[f"{name}, {GL_BULK}"] = kl_eddy(eps_b, nu, diff, c, n), f"bulk dissipation, {eps_src}"
    else:
        warn("--gas-liquid: no impeller torque and no dissipation in Fluid.txt; kL from the "
             "bulk dissipation skipped")
    if eps_s is not None:
        add(rel, GL_EPS_STATS, eps_s, "interface", "eps_surface [W/kg]")
        for name, c, n in KL_EDDY_MODELS:
            kls[f"{name}, {GL_STATS}"] = kl_eddy(eps_s, nu, diff, c, n), f"surface mean dissipation of {rel}"
    if u_s is not None:
        add(rel, "Free-Surface Speed, area mean (Stats) [m/s]", u_s, "interface", "u_surface [m/s]")
        if renewal:
            kls[f"Higbie, {GL_STATS}"] = kl_higbie(u_s, renewal, diff), f"surface mean speed of {rel}"
        if d_imp:
            kls[f"Thomas, {GL_STATS}"] = kl_thomas(u_s, d_imp, nu, diff), f"surface mean speed of {rel}"
    for name, (k, method) in kls.items():
        add(GL_FILE, f"kL {name} [m/s]", k, method)
    if a is not None:
        for name, (k, method) in kls.items():
            add(GL_FILE, f"kLa {name} [1/s]", k * a, method)
    if out_path is not None and len(cols) > 1:
        write_csv([dict(zip(cols, map(float, vals))) for vals in zip(*cols.values())], out_path,
                  columns=list(cols), what="time points")
    return rows, {"t": t, "vol": vol, "area": area}


def gas_liquid_assessment(rows: list[dict], gl_rows: list[dict], s: dict, dx: float | None,
                          diff: float, nu: float, t_start: float,
                          out_csv: Path | None = None) -> list[dict]:
    """What a calculation that needs no kL model would take, against the grid used.

    eps_s = dissipation at the free surface (local VTK value, else the surface mean of the
    interface Stats file, else the bulk dissipation), Sc = nu / D:
      eta_s = (nu^3 / eps_s)^(1/4)      Kolmogorov scale; a DNS of the surface turbulence
                                        needs dx <= 2.1 eta_s
      eta_B = eta_s Sc^(-1/2)           Batchelor scale, the finest concentration structure
      delta_c = D / kL                  thickness of the concentration film (kL: Lamont-Scott
                                        at eps_s, delta_c = 2.5 eta_B); resolved with
                                        CELLS_RESOLVED cells across it
    each with the refinement dx / dx_required, the number of cells and the lattice cost
    factor refinement^4 (cells x time steps). Also: whether the grid holds the surface
    shape (capillary length / dx) and bubbles entrained from it (Hinze d_max at eps_max /
    dx, closed structures of the VOF surface), and whether the liquid is mixed faster than
    the gas is transferred (blend-time correlation against t90 = ln 10 / kLa; HIM p. 628).
    Writes a readable table of the checks to out_csv."""
    pos = lambda x: isinstance(x, (int, float)) and np.isfinite(x) and x > 0  # noqa: E731
    gl = {r["variable"]: r["mean"] for r in gl_rows}
    by_var = {r["variable"]: r["mean"] for r in rows}
    out: list[dict] = []
    checks: list[dict] = []

    def add(var, val, note=GL_ASSESS):
        if isinstance(val, str) or pos(val):
            out.append(value_row(GL_ASSESS, var, val if isinstance(val, str) else float(val), note, t_start))
            log(f"  {var}: {val if isinstance(val, str) else f'{val:.4g}'}")

    def check(item, val, unit, verdict, basis):
        checks.append({"check": item, "value": val, "unit": unit, "assessment": verdict, "basis": basis})

    eps_b = gl.get(GL_EPS_BULK)
    eps_s, where = next(((gl[k], w) for k, w in ((GL_EPS_LOCAL, "local value of the VTK output"),
                                                (GL_EPS_STATS, "surface mean of the interface Stats file"))
                         if pos(gl.get(k))), (eps_b, "bulk dissipation (no surface value)"))
    sc = nu / diff
    n_cells = float(np.prod([s.get(f"L{a} [cells]") or np.nan for a in "XYZ"]))

    def required(what, dx_req, basis):
        """Rows and the check for one resolution requirement; returns the refinement."""
        add(f"Voxel Size for {what} [m]", dx_req, f"{GL_ASSESS}: {basis}")
        if not pos(dx):
            return None
        refine = dx / dx_req
        add(f"Grid Refinement for {what} [m/m]", refine, f"{GL_ASSESS}: voxel size used / voxel size required")
        if refine > 1:
            add(f"Lattice Cost Factor for {what} [count]", refine ** 4,
                f"{GL_ASSESS}: refinement^4 (cells x time steps)")
            if pos(n_cells):
                add(f"Lattice Cells for {what} [count]", n_cells * refine ** 3)
        return refine

    if pos(dx):
        add("Voxel Size Used dx [m]", dx, "input.xml")
    if pos(eps_b):
        add("Kolmogorov Scale at Bulk Dissipation [m]", (nu ** 3 / eps_b) ** 0.25)
    if pos(eps_s):
        eta = (nu ** 3 / eps_s) ** 0.25
        eta_b = eta * sc ** -0.5
        k_ref = kl_eddy(eps_s, nu, diff)
        delta = diff / k_ref
        add("Kolmogorov Scale at the Free Surface eta_s [m]", eta, f"{GL_ASSESS}: eps_s = {where}")
        add("Batchelor Scale at the Free Surface eta_B = eta_s Sc^-1/2 [m]", eta_b)
        add("Concentration Film Thickness delta_c = D/kL [m]", delta,
            f"{GL_ASSESS}: kL = Lamont-Scott at eps_s ({where})")
        r_t = required("Resolved Free-Surface Turbulence", DNS_DX_ETA * eta,
                       f"DNS: dx = {DNS_DX_ETA:g} eta_s")
        r_f = required("a Resolved Concentration Film", delta / CELLS_RESOLVED,
                       f"{CELLS_RESOLVED:g} cells across delta_c")
        if r_t is not None:
            add("Voxel Size / Kolmogorov Scale at the Free Surface [m/m]", dx / eta)
            turb = "resolved (DNS)" if r_t <= 1 else "modelled (LES): eddies below the voxel size are subgrid"
            add("Free-Surface Turbulence Resolution", turb)
            check("free-surface turbulence, dx / eta_s", dx / eta, "m/m", turb,
                  f"Kolmogorov scale eta_s = (nu^3/eps_s)^(1/4) at eps_s = {eps_s:.3g} W/kg ({where}); "
                  f"a DNS needs dx <= {DNS_DX_ETA:g} eta_s")
            check("voxel size for resolved free-surface turbulence", DNS_DX_ETA * eta, "m",
                  f"refine {r_t:.3g}x" + (f", lattice cost x{r_t ** 4:.3g}" if r_t > 1 else " (met)"),
                  "cost = refinement^4 (cells x time steps)")
            cells_f = delta / dx
            film = "resolved" if cells_f >= CELLS_RESOLVED else "marginal" if cells_f >= CELLS_MARGINAL \
                else "subgrid: kL has to come from a model (as here)"
            add("Concentration Film Thickness / Voxel Size [m/m]", cells_f)
            add("Concentration Film Resolution", film)
            check("concentration film, delta_c / dx", cells_f, "m/m", film,
                  f"delta_c = D/kL with the Lamont-Scott kL at eps_s = 2.5 eta_B, Sc = {sc:.3g}; "
                  f"resolved with {CELLS_RESOLVED:g} cells across, marginal with {CELLS_MARGINAL:g}")
            check("voxel size for a resolved concentration film", delta / CELLS_RESOLVED, "m",
                  f"refine {r_f:.3g}x" + (f", lattice cost x{r_f ** 4:.3g}" if r_f > 1 else " (met)")
                  + (f", {n_cells * max(r_f, 1.0) ** 3:.3g} cells" if pos(n_cells) else ""),
                  "a calculation that needs no kL model: surface turbulence and the film both resolved")

    # does the grid hold the area: surface shape and entrained gas
    sigma, rho = s.get("Surface tension [N/m]"), s.get("Fluid 1 density [kg/m^3]")
    if pos(sigma) and pos(rho) and pos(dx):
        cap = (sigma / (rho * G)) ** 0.5
        add("Capillary Length (sigma/(rho g))^0.5 [m]", cap)
        add("Capillary Length / Voxel Size [m/m]", cap / dx)
        check("surface shape, capillary length / dx", cap / dx, "m/m",
              "waves and the vortex are resolved; ripples shorter than a few voxels are not"
              if cap / dx >= CELLS_MARGINAL else "capillary ripples are below the grid: the area is a lower bound",
              f"capillary length (sigma/(rho g))^0.5; fewer than {CELLS_MARGINAL:g} voxels = not resolved")
    hinze = [(k, v) for k, v in by_var.items()
             if k.startswith("Hinze d_max at Maximum Dissipation / Voxel Size") and pos(v)]
    if hinze:
        cells_b = min(v for _, v in hinze)
        entr = "resolved" if cells_b >= CELLS_RESOLVED else "marginal" if cells_b >= CELLS_MARGINAL else "subgrid"
        add("Entrained Bubble Resolution", entr)
        check("entrained bubbles, Hinze d_max at eps_max / dx", cells_b, "m/m",
              entr + ("" if entr == "resolved" else ": gas drawn in from the surface would be lost below the "
                      "grid, the surface area is a lower bound once aeration starts"),
              "largest stable bubble at the impeller, d_max = 0.725 (sigma/rho)^0.6 eps_max^-0.4")
    share = gl.get("Closed-Structure Share of the Free-Surface Area (VOF surface) [m^2/m^2]")
    if isinstance(share, (int, float)) and np.isfinite(share):
        check("closed structures (bubbles, droplets) of the VOF surface", share, "m^2/m^2",
              "share of the free-surface area that is not the main surface",
              "closed pieces of the VOF surface; their area is part of A and of the kLa")

    # is the liquid mixed faster than the gas is transferred
    kla = [(k, v) for k, v in gl.items() if k.startswith("kLa ") and pos(v)]
    blend = [v for k, v in by_var.items() if k.startswith("Blend Time t95 Correlation") and pos(v)]
    if kla:
        k_hi, kla_hi = max(kla, key=lambda kv: kv[1])
        k_lo, kla_lo = min(kla, key=lambda kv: kv[1])
        add("Lowest kLa of the Models [1/s]", kla_lo, f"{GL_ASSESS}: {k_lo}")
        add("Highest kLa of the Models [1/s]", kla_hi, f"{GL_ASSESS}: {k_hi}")
        t90 = np.log(10.0) / kla_hi
        add("Mass Transfer Time t90 = ln10/kLa at the Highest kLa [s]", t90, f"{GL_ASSESS}: {k_hi}")
        if blend:
            ratio = min(blend) / t90
            mixing = "well mixed: one bulk concentration applies" if ratio < 0.1 \
                else "mixing and transfer on similar time scales: bulk gradients likely" if ratio < 1 \
                else "mixing-limited: solve the dissolved gas as a scalar with a free-surface flux in the CFD model"
            add("Blend Time / Mass Transfer Time t95/t90 [s/s]", ratio)
            add("Liquid Mixing vs Mass Transfer", mixing)
            check("blend time / mass transfer time", ratio, "s/s", mixing,
                  f"blend-time correlation HIM eq. (9-1) against t90 = ln 10 / kLa of the highest estimate ({k_hi}); "
                  "mixing time longer than the mass transfer time: HIM p. 628")
    if out_csv is not None and checks:
        write_csv(checks, out_csv, columns=["check", "value", "unit", "assessment", "basis"], what="checks")
    return out


def gas_liquid(case: Path, base: Path, files: list[Path], stats_dir: Path, rows: list[dict],
               s: dict, t_start: float, args, out_base: Path, case_id: str | None = None,
               workers: int = 1) -> list[dict]:
    """Free-surface gas-liquid mass transfer of one case (--gas-liquid free-surface): the
    specific free-surface area and kL / kLa from the Stats files (free_surface_stats), from
    the VTK output at the surface (free_surface_local, or the rows of an earlier run when
    the VTK files are gone) and the grid needed to resolve the transfer
    (gas_liquid_assessment). The case must use M-Star's free-surface model."""
    pos = lambda x: isinstance(x, (int, float)) and np.isfinite(x) and x > 0  # noqa: E731
    model = s.get("Fluid model")
    if model != FREE_SURFACE:
        fail(f"{case}: --gas-liquid free-surface needs M-Star's free-surface model; "
             + (f"this case uses the {model} model" if model
                else "no <fluidModelType> found (no readable input.xml)"))
    diff = args.gas_diffusivity
    nu = args.gas_nu if args.gas_nu is not None else s.get("Fluid 1 kinematic viscosity [m^2/s]")
    if not pos(nu):
        fail(f"{case}: --gas-liquid needs the kinematic viscosity of the liquid (not a number in "
             "input.xml); give --gas-nu")
    voxel = args.voxel_size or s.get("dx [m]")
    tank = s.get("Tank diameter T (domain width) [m]")
    renewal = args.renewal_length if args.renewal_length else 0.5 * tank if pos(tank) else None
    d_imp = max((v for k, v in s.items() if k.startswith("Impeller diameter (") and pos(v)), default=None)
    log(f"gas-liquid mass transfer (free surface): D = {diff:g} m^2/s, nu = {nu:g} m^2/s, "
        f"Sc = {nu / diff:.4g}"
        + (f", renewal length {renewal:.4g} m" if renewal else "")
        + (f", impeller diameter {d_imp:.4g} m" if d_imp else ""))
    param = lambda var, val, note: value_row(GL_FILE, var, float(val), note, t_start)  # noqa: E731
    out = [param("Gas Diffusivity D [m^2/s]", diff, "input"),
           param("Liquid Kinematic Viscosity [m^2/s]", nu, "input" if args.gas_nu is not None else "input.xml"),
           param("Schmidt Number Sc = nu/D [(m^2/s)/(m^2/s)]", nu / diff, "input")]
    if renewal:
        out.append(param("Surface Renewal Length L [m]", renewal,
                         "input" if args.renewal_length else "T/2 (half the tank diameter)"))
    if d_imp:
        out.append(param("Impeller Diameter for the Surface Reynolds Number [m]", d_imp, "input.xml (largest impeller)"))
    rows_st, series = free_surface_stats(base, files, stats_dir, s, t_start, diff, nu, renewal, d_imp,
                                         args.liquid_volume, args.area_column,
                                         out_base / "gas_liquid_timeseries.csv")
    if not series:
        fail(f"{case}: --gas-liquid needs the interface Stats file or Fluid.txt with the fluid volume")
    out.extend(rows_st)
    rows_loc: list[dict] = []
    if not args.gas_no_local:
        if not pos(voxel):
            warn("--gas-liquid: no voxel size (<dx> in input.xml or --voxel-size); local kL estimate skipped")
        else:
            vof_root = base
            if args.vof_dir is not None:
                vof_root = args.vof_dir if args.vof_dir.is_absolute() or not args.batch else base / args.vof_dir
            ok = np.isfinite(series["t"]) & np.isfinite(series["vol"])
            a_int = None
            if series["area"] is not None:
                fin = ok & np.isfinite(series["area"])
                a_int = (series["t"][fin], series["area"][fin])
            rows_loc = free_surface_local(vof_root, t_start, voxel, diff, nu,
                                          (series["t"][ok], series["vol"][ok]), renewal, d_imp, a_int,
                                          out_base, workers)
        if not rows_loc and args.reuse_results:
            rows_loc = reuse_results(args.reuse_results, case_id, GL_FILES, out_base)
    out.extend(rows_loc)
    if not any(r["variable"].startswith("kLa ") for r in out):
        warn(f"{case}: --gas-liquid: no kLa could be computed (no free-surface area in the Stats "
             "files and no VOF surface output)")
    log("gas-liquid assessment (grid needed to resolve the transfer, mixing):")
    out.extend(gas_liquid_assessment(rows, out, s, voxel if pos(voxel) else None, diff, nu, t_start,
                                     out_base / "gas_liquid_assessment.csv"))
    return out


# ---------------------------------------------------------------- plots

def flat_range(v: np.ndarray, rel: float = 1e-3) -> list[float] | None:
    """y-axis range of +-0.1% around a series that varies by less than 0.1% of its
    value, else None (autoscale); stops Plotly from zooming into float noise with
    20-digit ticks."""
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = float(v.min()), float(v.max())
    if hi - lo >= rel * max(abs(lo), abs(hi)):
        return None
    c = (lo + hi) / 2
    d = rel * abs(c) or 1.0
    return [c - d, c + d]


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
        name = fix_unit(header[j].strip())
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
            # float noise (e.g. 8e-17 on a constant volume) is shown as 0
            std_shown = std if abs(std) > 1e-9 * max(abs(mean), 1e-300) else 0.0
            fig.add_hrect(y0=mean - std, y1=mean + std, fillcolor="red",
                          opacity=0.1, line_width=0, row=r, col=c)
            fig.add_hline(y=mean, line_dash="dot", line_color="red", line_width=1,
                          annotation_text=f"{mean:.4g} \u00b1 {std_shown:.3g}",
                          annotation_font_size=10, row=r, col=c)
        fig.update_xaxes(title_text="Time [s]", title_font_size=10, row=r, col=c)
        fig.update_yaxes(title_text=var_unit(name), title_font_size=10, tickformat=".6~g",
                         range=flat_range(v), row=r, col=c)

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
                             "normalized a")], True, "(1/m)/(1/m)", marks))
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
                        for tag, label in phases], True, "m^3/m^3"))
        panels.append(("Number of drops n_drops(t)",
                       [line(vof["t"], vof[f"n_drops_{tag}"], label)
                        for tag, label in phases], True, "count"))
    if np.isfinite(vof.get("coverage", [np.nan])).any():
        panels.append(("Drop interface coverage A_drops/A_int (t)",
                       [line(vof["t"], vof["coverage"], "coverage")], True, "m^2/m^2"))
    if uni.get("combos"):
        panels.append(("Zonal CoV of phase fraction (t)",
                       [line(uni["t"], uni[f"cov{key}"], f"CoV {label}")
                        for key, label in uni["combos"]], True, "m^3/m^3"))
        s_lines = [line(uni["t"], uni[f"S{key}"], f"S {label}")
                   for key, label in uni["combos"] if f"S{key}" in uni]
        if s_lines:
            panels.append(("Segregation index S(t) (0 homogeneous, 1 segregated)",
                           s_lines, True, "m^3/m^3"))
    if uni.get("share"):
        panels.append(("Largest connected region share (t)",
                       [line(uni["share_t"], v, label) for label, v in uni["share"].items()],
                       True, "m^3/m^3"))

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
                       False, "count/count"))

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
                       colorbar=dict(title="\u03c6 [m^3/m^3]", len=0.3, y=0.15),
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
                 plot_dir: Path | None = None, out_base: Path | None = None,
                 case_id: str | None = None
                 ) -> tuple[list[dict], dict[str, list[dict]]]:
    """Summarize one case; write its CSVs (and plots) to out_base (default: the
    case dir). Returns (all rows, {"dispersion": rows, "uniformity": rows,
    "gas_liquid": rows}); ([], {}) on failure."""
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
            fluid = [f for f in files if f.name.lower() == "fluid.txt"]
            table = load_stats_table(fluid[0]) if fluid else None
            t_all = table[1][:, 0] if table is not None else np.array([])
            if not np.isfinite(t_all).any():
                warn(f"{case}: could not detect a mean-velocity plateau; "
                     "specify --time instead")
                return [], {}
            t_start, method = fallback_start(t_all, args.window), FALLBACK
            warn(f"{case}: no mean-velocity plateau (flow not steady); averaging the "
                 f"last part of the run, t >= {t_start:.3f} s (specify --time to override)")
        else:
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
    settings = case_settings(case, stats_dir)
    rows.extend(power_metrics(base, files, rows, settings, t_start))
    out_base = out_base or base
    out_base.mkdir(parents=True, exist_ok=True)
    disp: dict = {}
    disp_rows: list[dict] = []  # evaluated from t_disp
    groups: dict[str, list[dict]] = {"dispersion": [], "uniformity": [], "gas_liquid": []}
    t_disp, disp_method = t_start, None
    workers = getattr(args, "workers", 1)
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
        kin = disp["interface"].get("kinetics") or {}
        if np.isfinite(kin.get("t_ss_a", np.nan)):
            t_disp, disp_method = max(t_start, kin["t_ss_a"]), "a(t) plateau"
        elif np.isfinite(kin.get("t_ref", np.nan)):
            # no plateau: the window the a(t) kinetics fall back to, not the whole transient
            # (6 digits: the first Stats sample is at dt, which would push the window start
            # just past the output frame written at exactly 75 % of the run)
            t_disp, disp_method = max(t_start, float(f"{kin['t_ref']:.6g}")), NO_PLATEAU
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
                                                   t_disp, fluids, a_int, out_dir=out_base,
                                                   min_d_dx=args.min_d_dx, workers=workers)
            if not rows_vof and args.reuse_results:
                rows_vof = reuse_results(args.reuse_results, case_id, VOF_FILES, out_base)
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
        cont = re.match(r"Fluid (\d+)", str(settings.get("Continuous phase (largest initial volume)", "")))
        sigma = args.sigma if args.sigma is not None else settings.get("Surface tension [N/m]")
        rho_c = args.rho_continuous if args.rho_continuous is not None \
            else settings.get(f"Fluid {cont.group(1) if cont else 1} density [kg/m^3]")
        rows_res = resolution_check(
            files, t_disp, voxel, sigma, rho_c, args.hinze_c,
            args.k_phi, args.eps_ratio, args.eps_file, phi_d)
        disp_rows.extend(rows_res)
        groups["dispersion"].extend(rows_res)
        rows_mt = mass_transfer(rows, settings, disp["interface"].get("kinetics") or {}, phi_d,
                                args.diff_c, args.diff_d, args.partition, args.dc0, t_disp, coverage)
        disp_rows.extend(rows_mt)
        groups["dispersion"].extend(rows_mt)
        dispersion_timeseries(
            disp["interface"], out_base / "dispersion_timeseries.csv", files, voxel,
            sigma, rho_c, args.hinze_c, args.k_phi, args.eps_ratio,
            args.eps_file, phi_d)
    if args.uniformity:
        scales = args.zone_scales or [(args.zones_z, args.zones_r)]
        rows_uni, disp["uniformity"] = phase_uniformity(
            base, args.zones_z, args.zones_r, t_disp, args.vertical_axis,
            args.axis_center, out_dir=out_base, zone_scales=scales, window=args.window,
            tol=args.tol, continuity_threshold=args.continuous_threshold, workers=workers)
        if not rows_uni and args.reuse_results:
            rows_uni = reuse_results(args.reuse_results, case_id, UNI_FILES, out_base)
        disp_rows.extend(rows_uni)
        groups["uniformity"].extend(rows_uni)
    if args.dispersion and len(fluids) == 2:
        log("dispersion assessment (grid resolution, correlations, area bounds):")
        rows_as = dispersion_assessment(
            rows + disp_rows, settings, disp["interface"].get("kinetics") or {}, phi_d, voxel,
            t_disp, args.min_d_dx, sigma, out_base / "dispersion_assessment.csv",
            args.diff_c, args.diff_d, args.partition)
        disp_rows.extend(rows_as)
        groups["dispersion"].extend(rows_as)
    if disp_method:
        inner = disp_method.replace(" (", ", ").rstrip(")")  # no brackets within brackets
        for r in disp_rows:
            if r["file"] in ("mass transfer", "dispersion assessment"):
                r["steady_method"] = f"{r['steady_method']} ({inner})"
                r["steady_start_s"] = t_disp
            elif not any(k in str(r["steady_method"]) for k in ("fallback", "no plateau", "reused")):
                r["steady_method"], r["steady_start_s"] = disp_method, t_disp
    rows.extend(disp_rows)
    if getattr(args, "gas_liquid", None):
        groups["gas_liquid"] = gas_liquid(case, base, files, stats_dir, rows, settings, t_start, args,
                                          out_base, case_id, workers)
        rows.extend(groups["gas_liquid"])
    if method == FALLBACK:
        for r in rows:
            if r["steady_start_s"] == t_start and "fallback" not in str(r["steady_method"]):
                r["steady_method"] = f"{r['steady_method']} ({FALLBACK})"

    out = output or out_base / "stats_summary.csv"
    addons_only = getattr(args, "addons_only", False)
    if not addons_only:
        write_csv(rows, out)
    kept = {id(r) for r in rows}  # the area-based d32 may have been dropped
    groups = {k: [r for r in g if id(r) in kept] for k, g in groups.items()}
    for name, enabled in (("dispersion", args.dispersion), ("uniformity", args.uniformity),
                          ("gas_liquid", getattr(args, "gas_liquid", None))):
        if not enabled:
            continue
        if groups[name]:
            write_csv(groups[name], out.with_name(f"{out.stem}_{name}{out.suffix}"))
        else:
            warn(f"--{name}: no metrics computed; no {name} summary written")

    if args.plot is not None and HTML_PLOTS:
        plot_dir = plot_dir or out_base / "stats_plots"
        if not addons_only:
            make_plots(files, stats_dir, args.plot, t_start, plot_dir, mix_vlines=mix_vlines)
        if disp:
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_dispersion(disp, t_disp, plot_dir / "dispersion.html")
    return rows, groups


def case_dir_of(stats_dir: Path, top: Path) -> Path:
    """Case folder of a Stats dir: the nearest ancestor (up to top) holding
    input.xml or a .msb file, else the Stats parent (its parent for M-Star's out/)."""
    for d in [stats_dir.parent, *stats_dir.parent.parents]:
        if (d / "input.xml").is_file() or any(d.glob("*.msb")):
            return d
        if d == top:
            break
    p = stats_dir.parent
    return p.parent if p.name.lower() == "out" and p != top else p


def find_batch_cases(parent: Path) -> list[tuple[Path, Path]]:
    """Return (case dir, stats dir) for cases under parent, at any depth."""
    cases_by_dir: dict[Path, Path] = {}
    for sub in sorted(p for p in parent.iterdir()
                      if p.is_dir() and not p.name.startswith(".")):
        hits = sorted(p for p in sub.rglob("Stats")
                      if p.is_dir() and not is_hidden(p, sub))
        if hits:
            for stats_dir in hits:
                case = case_dir_of(stats_dir, sub)
                if case not in cases_by_dir or stats_rank(case, stats_dir) < stats_rank(case, cases_by_dir[case]):
                    cases_by_dir[case] = stats_dir
        else:
            stats_dir = locate_stats_dir(sub)
            if stats_dir is not None:
                cases_by_dir[sub] = stats_dir
    return sorted(cases_by_dir.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("case", type=Path,
                    help="M-Star case directory, or the Stats directory itself; "
                         "with --batch, a parent directory of case subfolders")
    ap.add_argument("--batch", action="store_true",
                    help="treat CASE as a parent directory and recursively "
                        "process case subfolders containing stats; writes one "
                        "CSV per case plus a combined batch CSV")
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
    ap.add_argument("--migrate-units", action="store_true",
                    help="only rewrite existing result CSVs under CASE (a results "
                         "folder) with the current unit labels, then exit")
    ap.add_argument("--rename-body", nargs="?", const=BODY_CANON, default=None, metavar="NAME",
                    help="only rename the impeller body ('Moving Body', 'agitator') to NAME "
                         f"(default {BODY_CANON}) in existing results under CASE (a results "
                         "folder: CSV headers and the cells of the file / variable columns, "
                         "merging columns the rename makes equal; HTML text; file names), "
                         "then exit")
    ap.add_argument("--plot", nargs="*", metavar="VAR", default=None,
                    help="(with --plot-html) write one interactive HTML of plots per "
                         "stats file; give variable names (fuzzy) to limit which are "
                         "plotted, or no names to plot all variables")
    ap.add_argument("--plot-html", action="store_true",
                    help="write the legacy HTML figures (means bar plots, --plot Stats "
                         "plots, dispersion.html); off by default because every result "
                         "is in the CSVs and the M-Star Post app plots from those")
    ap.add_argument("--plot-dir", type=Path, default=None,
                    help="plot output directory (default: <case>/stats_plots; "
                         "ignored with --batch)")
    ap.add_argument("--output-root", type=Path, default=None, metavar="DIR",
                    help="write all outputs under DIR instead of the case "
                         "folders (with --batch, per-case outputs go to "
                         "DIR/<case path relative to CASE>/), leaving the "
                         "case folders untouched")
    ap.add_argument("--cases", nargs="+", default=None, metavar="REL",
                    help="with --batch, only process these cases (paths "
                         "relative to CASE, as listed in the batch CSVs)")

    dg = ap.add_argument_group(
        "phase dispersion (Immiscible Two-Fluid; VTK parts need pyvista)",
        "example: mstar_stats_summary.py case --dispersion --dispersed-fraction 0.2 "
        "--voxel-size 5e-4 --sigma 0.05 --rho-continuous 998 --plot")
    dg.add_argument("--addons-only", action="store_true",
                    help="with --dispersion/--uniformity: write only their outputs (the "
                         "*_dispersion/*_uniformity summaries, per-case time series and "
                         "dispersion.html), not the base Stats summary, means, settings or "
                         "Stats plots; for running the add-on after a separate base summary")
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
                         "(default: max(flow steady state, a(t) plateau start); without an "
                         "a(t) plateau the last max(WINDOW, 25%%) of the a(t) trace)")
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
    dg.add_argument("--diff-c", type=float, default=None, metavar="M2S",
                    help="solute diffusivity in the continuous phase [m^2/s]: enables the "
                         "mass-transfer estimate (two-film, global averages)")
    dg.add_argument("--diff-d", type=float, default=None, metavar="M2S",
                    help="solute diffusivity in the dispersed phase [m^2/s]")
    dg.add_argument("--partition", type=float, default=None, metavar="M",
                    help="equilibrium distribution coefficient m = C_d*/C_c* (needed for "
                         "K_oc a and the extraction time)")
    dg.add_argument("--dc0", type=float, default=None, metavar="MOL_M3",
                    help="initial driving force C_c - C_d/m [mol/m^3] for the initial "
                         "transfer rate N0 [mol/s]")
    dg.add_argument("--eps-ratio", type=float, default=1.0,
                    help="local/mean energy dissipation ratio in the Hinze check, "
                         "e.g. impeller-region eps / mean eps")
    dg.add_argument("--eps-file", default=None, metavar="FILE",
                    help="stats file (path or name fragment, e.g. an impeller "
                         "ControlVolume) to take eps from for the Hinze check "
                         "(default: Fluid.txt)")
    dg.add_argument("--reuse-results", nargs="+", type=Path, default=None, metavar="RUN_DIR",
                    help="earlier output roots (newest first): VOF-drop and uniformity "
                         "metrics whose .vtp/.vti files are gone are taken from the first one "
                         "holding them (rows and per-case CSVs; with --batch the rows of the "
                         "case in batch_summary*.csv, else the root's stats_summary*.csv); "
                         "everything else is recomputed. Default with --output-root "
                         "<...>_dispersion: the earlier sibling *_dispersion folders "
                         "(mstar_post runs), newest first")
    dg.add_argument("--no-reuse", action="store_true", help="never reuse earlier results")
    dg.add_argument("--workers", type=int, default=1, metavar="N",
                    help="processes analysing the VTK frames (.vtp drop statistics, .vti "
                         "uniformity) side by side; the results equal --workers 1. Each "
                         "worker holds one frame in memory (about 35 bytes per voxel, or "
                         "0.5 GB per million triangles of the VOF surface if that is more)")
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
    gg = ap.add_argument_group(
        "gas-liquid mass transfer (M-Star free-surface model; the local estimate needs pyvista)",
        "example: mstar_stats_summary.py case --gas-liquid free-surface --gas-diffusivity 2e-9")
    gg.add_argument("--gas-liquid", nargs="?", const="free-surface", choices=["free-surface"],
                    default=None, metavar="ROUTE",
                    help="liquid-side gas-liquid mass transfer; ROUTE free-surface (the only one): "
                         "transfer through the free surface alone, no sparged or entrained gas. "
                         "Specific free-surface area a = A/V_liq and kL, kLa of several models "
                         "(Lamont-Scott, Kawase-Moo-Young, Kastanek, rigid surface, Higbie, Thomas, "
                         "surface divergence) from the Stats files (gas_liquid_timeseries.csv) and "
                         "from the fields at the surface in the VTK output (gas_liquid_surface.csv), "
                         "plus the grid a calculation without a kL model would need "
                         "(gas_liquid_assessment.csv). Needs --gas-diffusivity and a case run "
                         "with M-Star's free-surface model; uses --liquid-volume, --area-column, "
                         "--vof-dir, --voxel-size, --workers and --reuse-results as above")
    gg.add_argument("--gas-diffusivity", type=float, default=None, metavar="M2S",
                    help="diffusivity of the dissolved gas in the liquid [m^2/s] (e.g. O2 in water "
                         "at 25 C: 2.1e-9)")
    gg.add_argument("--gas-nu", type=float, default=None, metavar="M2S",
                    help="kinematic viscosity of the liquid [m^2/s] (default: input.xml)")
    gg.add_argument("--renewal-length", type=float, default=None, metavar="M",
                    help="length L [m] a liquid element travels along the surface before it is "
                         "replaced, for the Higbie contact time t_c = L / u_s (default: half the "
                         "tank diameter)")
    gg.add_argument("--gas-no-local", action="store_true",
                    help="skip the local estimate from the VTK output (Stats files only)")
    args = ap.parse_args(argv)
    global HTML_PLOTS
    HTML_PLOTS = bool(args.plot_html)
    if args.plot_html and args.plot is None and not (args.migrate_units or args.rename_body):
        args.plot = []  # --plot-html alone: plot every variable

    if args.migrate_units:
        if not args.case.is_dir():
            fail(f"--migrate-units requires a directory: {args.case}")
        log(f"{migrate_units(args.case, args.ref_case)} file(s) updated")
        return
    if args.rename_body:
        if not args.case.is_dir():
            fail(f"--rename-body requires a directory: {args.case}")
        log(f"{migrate_body_names(args.case, args.ref_case, args.rename_body)} file(s) updated")
        return
    if args.dispersed_fraction is not None and not 0 < args.dispersed_fraction < 1:
        fail("--dispersed-fraction must be between 0 and 1")
    if args.no_reuse:
        args.reuse_results = None
    elif args.reuse_results is None and args.output_root and args.output_root.parent.is_dir():
        # mstar_post run folders are <time stamp>_<task>: the earlier runs of the same task
        kind = next((k for k in ("_dispersion", "_gas_liquid") if args.output_root.name.endswith(k)), None)
        earlier = sorted((d for d in args.output_root.parent.glob(f"*{kind}")
                          if d.is_dir() and d.resolve() != args.output_root.resolve()),
                         reverse=True) if kind else []
        args.reuse_results = earlier or None
    if args.workers < 1:
        fail("--workers must be >= 1")
    if args.addons_only and not (args.dispersion or args.uniformity or args.gas_liquid):
        fail("--addons-only needs --dispersion, --uniformity and/or --gas-liquid")
    if args.gas_liquid:
        if args.gas_diffusivity is None or not args.gas_diffusivity > 0:
            fail("--gas-liquid needs --gas-diffusivity > 0 [m^2/s]")
        if args.gas_nu is not None and not args.gas_nu > 0:
            fail("--gas-nu must be > 0")
        if args.renewal_length is not None and not args.renewal_length > 0:
            fail("--renewal-length must be > 0")
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
        ids = {c: c.relative_to(args.case).as_posix() for c, _ in cases}
        if args.cases:
            wanted = {Path(c).as_posix().strip("/") for c in args.cases}
            for c in sorted(wanted - set(ids.values())):
                warn(f"--cases: '{c}' is not a case under {args.case}")
            cases = [(c, sd) for c, sd in cases if ids[c] in wanted]
            if not cases:
                fail("--cases matched no cases")
        log(f"batch mode: {len(cases)} case(s) under {args.case}")
        if args.output_root:
            args.output_root.mkdir(parents=True, exist_ok=True)

        batch_rows = []
        batch_groups: dict[str, list[dict]] = {}
        empty, failed = [], []
        for case, stats_dir in cases:
            cid = ids[case]
            log(f"=== case: {cid} ===")
            try:
                rows, groups = process_case(
                    case, stats_dir, args,
                    out_base=args.output_root / cid if args.output_root else None, case_id=cid)
            except (Exception, SystemExit) as e:  # one bad case must not cost the others their tables
                if not isinstance(e, SystemExit):
                    import traceback
                    traceback.print_exc()
                warn(f"{cid}: failed ({e if isinstance(e, SystemExit) else f'{type(e).__name__}: {e}'}); "
                     "continuing with the other cases")
                failed.append(cid)
                continue
            if not rows:
                empty.append(cid)
            for r in rows:
                batch_rows.append({"case": cid, **r})
            for name, g in groups.items():
                batch_groups.setdefault(name, []).extend({"case": cid, **r} for r in g)
        if (empty or failed) and batch_rows:
            warn(f"{len(empty) + len(failed)} of {len(cases)} case(s) are missing from the tables: "
                 + "; ".join(f"{what}: {', '.join(ids_)}" for what, ids_ in
                             (("failed", failed), ("no results", empty)) if ids_))
        if not batch_rows:
            fail("no variables summarized in any case")
        out = args.output or (args.output_root or args.case) / "batch_summary.csv"
        if not args.addons_only:
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
        if not args.addons_only:
            write_settings_csv([case_settings(c, sd, ids[c]) for c, sd in cases],
                               out.with_name(f"{out.stem}_settings{out.suffix}"))
    else:
        stats_dir = find_stats_dir(args.case)
        base = args.case if args.case.is_dir() else args.case.parent
        out = args.output or (args.output_root or base) / "stats_summary.csv"
        rows, groups = process_case(args.case, stats_dir, args, output=out,
                                    plot_dir=args.plot_dir, out_base=args.output_root)
        if not rows:
            fail("no variables summarized")
        if not args.addons_only:
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
        if not args.addons_only:
            write_settings_csv([case_settings(args.case, stats_dir)],
                               out.with_name(f"{out.stem}_settings{out.suffix}"))
    log("done")


if __name__ == "__main__":
    main()
