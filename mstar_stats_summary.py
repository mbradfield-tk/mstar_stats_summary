#!/usr/bin/env python3
"""Steady-state summary of all variables tracked in an M-Star Stats folder.

Scans every tab-delimited stats file (recursively, e.g. Fluid.txt,
MovingBody_*.txt, ControlVolume_*/FieldData.txt), detects a single steady-state
start time from the plateau of the mean fluid velocity in Fluid.txt
(windowed-mean drift criterion), or uses a user-defined start time, and writes
a CSV with the steady-state average and standard deviation of every variable.

Example:
    python mstar_stats_summary.py test              # auto-detect steady state
    python mstar_stats_summary.py test --time 5.0   # steady state = t >= 5 s
    python mstar_stats_summary.py test --plot "mean velocity" "power number"
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

def find_stats_dir(root: Path) -> Path:
    """Accept a case dir, a dir containing Stats/, or the Stats dir itself."""
    if not root.exists():
        fail(f"path not found: {root}")
    if root.is_dir():
        if root.name.lower() == "stats":
            return root
        hits = sorted(p for p in root.rglob("Stats") if p.is_dir())
        if hits:
            return hits[0]
        if any(root.glob("*.txt")):
            return root
    fail(f"no Stats folder (or .txt stats files) found under {root}")


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


def write_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {out_path} ({len(rows)} variables)")


# ---------------------------------------------------------------- plots

def make_plots(files: list[Path], stats_dir: Path, queries: list[str],
               t_start: float, out_dir: Path) -> None:
    """Plot the time course of every variable matching a query (fuzzy, unit-free)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qn = [q.strip().lower() for q in queries]
    matched = {q: 0 for q in qn}
    out_dir.mkdir(parents=True, exist_ok=True)
    n_plots = 0

    for f in files:
        table = load_stats_table(f)
        if table is None:
            continue
        header, data = table
        rel = str(f.relative_to(stats_dir))
        times = data[:, 0]
        for j in range(1, len(header)):
            name = header[j].strip()
            hits = [q for q in qn if q == norm_name(name) or q in norm_name(name)]
            if not hits:
                continue
            for q in hits:
                matched[q] += 1
            vals = data[:, j]
            ok = np.isfinite(times) & np.isfinite(vals)
            t, v = times[ok], vals[ok]
            if t.size == 0:
                continue
            steady = t >= t_start
            mean = float(v[steady].mean()) if steady.any() else np.nan
            std = float(v[steady].std(ddof=1)) if steady.sum() > 1 else 0.0

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(t, v, lw=1, color="steelblue")
            ax.axvline(t_start, color="tab:green", ls="--",
                       label=f"steady-state start @ {t_start:.2f} s")
            if np.isfinite(mean):
                ax.axhline(mean, color="tab:red", ls=":",
                           label=f"steady mean = {mean:.4g} \u00b1 {std:.3g}")
                ax.axhspan(mean - std, mean + std, color="tab:red", alpha=0.1)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(name)
            ax.set_title(f"{rel}: {name}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            stem = re.sub(r"[^\w.-]+", "_", f"{Path(rel).with_suffix('')}_{norm_name(name)}")
            fig.savefig(out_dir / f"{stem}.png", dpi=150)
            plt.close(fig)
            n_plots += 1

    for q in qn:
        if matched[q] == 0:
            warn(f"--plot '{q}' matched no variables")
    log(f"wrote {n_plots} plot(s) to {out_dir}")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("case", type=Path,
                    help="M-Star case directory, or the Stats directory itself")
    ap.add_argument("--time", type=float, default=None,
                    help="user-defined steady-state start time [s]; "
                         "overrides automatic detection")
    ap.add_argument("--window", type=float, default=1.0,
                    help="steady-state detection window [s]")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="steady-state tolerance on windowed-mean drift")
    ap.add_argument("--output", type=Path, default=None,
                    help="output CSV path (default: <case>/stats_summary.csv)")
    ap.add_argument("--plot", nargs="+", metavar="VAR", default=None,
                    help="plot the time course of variables matching these names "
                         "(fuzzy, e.g. 'mean velocity', 'power number')")
    ap.add_argument("--plot-dir", type=Path, default=None,
                    help="plot output directory (default: <case>/stats_plots)")
    args = ap.parse_args(argv)

    stats_dir = find_stats_dir(args.case)
    files = find_stats_files(stats_dir)
    if not files:
        fail(f"no .txt stats files found in {stats_dir}")
    log(f"stats folder: {stats_dir} ({len(files)} files)")

    if args.time is not None:
        t_start, method = args.time, "user"
        log(f"user-defined steady-state start: t >= {t_start} s")
    else:
        t_start = steady_from_fluid_velocity(files, args.window, args.tol)
        method = "auto"
        if t_start is None:
            fail("could not detect a mean-velocity plateau; specify --time instead")
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
        fail("no variables summarized")

    out = args.output or (args.case if args.case.is_dir() else args.case.parent) / "stats_summary.csv"
    write_csv(rows, out)

    if args.plot:
        plot_dir = args.plot_dir or (args.case if args.case.is_dir() else args.case.parent) / "stats_plots"
        make_plots(files, stats_dir, args.plot, t_start, plot_dir)
    log("done")


if __name__ == "__main__":
    main()
