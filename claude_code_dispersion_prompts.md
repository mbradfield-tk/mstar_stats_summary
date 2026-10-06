# Claude Code prompts: add phase-dispersion metrics to mstar_stats_summary.py

Run these in order, one at a time. Review and test after each step before moving on.

---

## 0. Context (send first)

```
Read mstar_stats_summary.py. I want to add phase-dispersion metrics for M-Star CFD
Immiscible Two-Fluid simulations. Keep the existing structure and conventions:
the log/warn/fail helpers, the row-dict schema in COLUMNS, Plotly HTML output,
--batch support, and graceful skipping when inputs are missing. All new features
must be opt-in, and the script must behave exactly as before when they're unused.
numpy stays the only hard dependency; import pyvista lazily inside the functions
that need it and warn and skip if it isn't installed. Don't write any code yet;
summarize your plan first.
```

## 1. Interface stats file: specific area and area-based d32

```
Add a function interface_metrics(files, stats_dir, t_start, phi, v_liq) that finds
the M-Star interface statistics file (ASCII, name contains "interface",
case-insensitive) among the stats files and loads it with load_stats_table.
Find the total interfacial area column with fuzzy matching via norm_name
(e.g. contains "area"). If it isn't found, warn and list the available headers.
Compute these time series:
  - specific interfacial area a(t) = A_int / V_liq  [1/m]
  - area-based Sauter diameter d32(t) = 6*phi / a  [m]
V_liq comes from --liquid-volume [m^3], or from the "Fluid Volume" column in
Fluid.txt if the flag isn't given. phi (dispersed-phase volume fraction) comes
from a new --dispersed-fraction argument; skip d32 with a warning if it isn't
given. Add steady-state mean/std rows for a and d32 (with t >= t_start) to the
summary using the existing row schema, with steady_method="interface". Enable it
with a new flag --dispersion.
```

## 2. VOF structure post-processing: drop size distribution and dispersed fraction

```
Add a function vof_structures(case_dir, min_voxels, voxel_size) that finds the
Volume VOF Surface .vtp files in the case output (search recursively; filename
contains "VOF" and ends in .vtp). Sort them by time; get the time from the
filename or from the file's field data, and tell me which you used. For each
frame:
  1. Read it with pyvista. If the file contains a structure/region ID array
     written by M-Star's "Identify VOF Structures" option, use it. Otherwise
     label regions with surf.connectivity(extraction_mode="all").
  2. For each region, extract_surface() and keep only closed surfaces
     (n_open_edges == 0); count clipped or open regions separately.
  3. Volume V_i = abs(volume), area A_i, equivalent diameter
     d_i = (6 V_i / pi)^(1/3).
  4. Drop structures with V_i < min_voxels * voxel_size^3 as numerical debris.
     Defaults are min_voxels=8; voxel_size comes from --voxel-size [m]. If it
     isn't given, warn and keep all structures.
Per frame, compute: n_drops, d32 = 6*sum(V)/sum(A), d10, d43, d_max,
d32_excl_largest (same as d32 but excluding the largest structure), and
dispersed fraction F_disp = 1 - V_largest/V_total. Write a time-series CSV
(vof_structures.csv) in the case folder, plus a per-frame drop-diameter CSV
(vof_dsd.csv: time, region_id, d_eq, volume, area). Report steady-state
mean/std of each metric (frames with t >= t_start) as summary rows with
steady_method="vof". Enable with --dispersion and the new --vof-dir (optional
override of the search path), --voxel-size, and --min-voxels arguments.
```

## 3. Spatial uniformity: zonal CoV and vertical holdup profile

```
Add an optional function phase_uniformity(case_dir, n_zones_z, n_zones_r) that
reads the 3D volume output (.vti) containing the phase-fraction/VOF field (find
the array by fuzzy name match on "vof", "phase" or "fraction"; warn and list the
arrays if it's ambiguous). For each frame:
  - Coarse-grain phi into zones BEFORE computing any statistics (voxel-level phi
    is ~0/1 and measures interface sharpness, not dispersion). Use n_zones_z
    axial bins x n_zones_r radial bins, with the vessel axis along z and the
    center at the domain x-y center (add --axis-center x y to override).
    Mask out non-fluid voxels if a solid or flag array exists.
  - Zonal CoV = std(phi_zone)/mean(phi_zone), volume-weighted.
  - Vertical holdup profile phi(z), averaged over horizontal slices.
Write phase_uniformity.csv (time, CoV) and holdup_profile.csv (time, z,
phi_mean), and add steady-state CoV mean/std to the summary rows with
steady_method="zonal". Enable with --uniformity and its arguments
(--zones-z default 10, --zones-r default 3).
```

## 4. Plots

```
Extend the Plotly output (only when --plot is also given) with a dispersion.html
in the stats_plots folder that contains: a(t), d32(t) from both methods
(interface-based and VOF-structure-based) on one axis, F_disp(t), n_drops(t),
zonal CoV(t), a DSD histogram (number and volume weighted) pooled over
steady-state frames, and a heatmap of the holdup profile phi(z, t). Add the same
green dashed steady-state start line used in plot_file. Skip any panel whose
data isn't available.
```

## 5. Batch integration

```
Make sure all new metrics flow into the --batch outputs. Add "interfacial area",
"d32", "dispersed fraction", "n_drops" and "cov" to DEFAULT_MEANS_VARS so they
appear in batch_summary_means.csv and its bar-chart HTML. Check that
DEFAULT_MEANS_EXCLUDE doesn't accidentally drop them (e.g. d_max contains
"max"; rename it to "largest drop diameter" or adjust the regex).
```

## 6. Resolution check

```
Add a resolution sanity check, run when --dispersion and --voxel-size are given:
estimate the Hinze maximum stable drop size
d_max_hinze = C * (sigma/rho_c)^0.6 * eps^(-0.4), with C=0.725 by default. eps
is the steady-state mean energy dissipation from Fluid.txt (fuzzy match "energy
dissipation"; prefer the per-unit-mass value in W/kg). sigma [N/m] and rho_c
[kg/m^3] come from new --sigma and --rho-continuous arguments. Report
d_max_hinze / voxel_size. If it's below 5, warn that drop sizes are under-resolved
and that the DSD/d32 results reflect grid scale (macro-dispersion metrics like
F_disp and holdup are still meaningful). Add the ratio as a summary row.
```

## 7. Tests and docs

```
Write pytest tests using synthetic data: (a) a fake interface stats .txt with a
known area series; (b) a .vtp built with pyvista from N spheres of known radii
plus one large slab, checking that d32, F_disp, n_drops and the min-voxel filter
match the analytical values; (c) a synthetic .vti phi field with a known
vertical gradient for the CoV/holdup code. Then update the module docstring and
the argparse help with example commands, e.g.
  python mstar_stats_summary.py case --dispersion --dispersed-fraction 0.2 \
      --voxel-size 5e-4 --sigma 0.05 --rho-continuous 998 --plot
```
