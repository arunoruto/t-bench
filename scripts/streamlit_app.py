"""
t-bench Dashboard.

Upload a cluster position file, pick a material (fixed refractive index
or a refidxdb-backed dispersive lookup) and a wavelength range, pick
which of the four adapters to run, and compare MSTM against FaSTMM2 --
both their Python wrappers and their CLI binaries -- on accuracy (cross-
section spectra) and speed (wall-clock time per wavelength) side by side.

No sidebar (a sidebar is a single global container regardless of which
part of the page you're looking at -- see the note in pyMSTM's and
pyFaSTMM's own dashboards for why that caused real bleed-through bugs
there); settings live in a row above a divider, full-width plots below.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from refidxdb import DATABASES

from tbench import ALL_ADAPTERS, MaterialSpec, SweepRequest, load_positions, run_sweep
from tbench.adapters.fastmm2_cli import Fastmm2CliAdapter

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """refractiveindex.info's catalog names use HTML for chemical-formula
    subscripts (e.g. "SiO<sub>2</sub>") -- strip tags for plain-text display."""
    return _HTML_TAG_RE.sub("", text)


@st.cache_data(show_spinner="Loading catalog...")
def _load_catalog(db_name: str):
    return DATABASES[db_name].catalog()

# One color per *tool* (not per adapter) -- the Python wrapper and the
# CLI binary of the same tool share a color, python drawn as a line, CLI
# as same-colored markers, so the two are easy to pair up visually. Same
# convention as pyMSTM's/pyFaSTMM's own dashboards.
_TOOL_COLORS = {"mstm": "#1f77b4", "fastmm2": "#d62728"}
_QUANTITY_LABELS = [("c_ext", "Cext"), ("c_abs", "Cabs"), ("c_sca", "Csca")]

st.set_page_config(page_title="t-bench Dashboard", layout="wide")
st.title("t-bench Dashboard")
st.caption(
    "Compare MSTM and FaSTMM2 -- Python wrapper and CLI, both tools -- on "
    "the same cluster across a wavelength range: accuracy and speed, side by side."
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

st.subheader("Cluster")
c_cluster1, c_cluster2 = st.columns([2, 1])
with c_cluster1:
    uploaded = st.file_uploader(
        "Position file (x, y, z, radius per line -- .dat/.txt/.csv, `#` comments OK)",
        type=["dat", "txt", "csv", "pos"],
    )
with c_cluster2:
    geom_scale = st.number_input(
        "Scale (multiplies every column)", value=1.0, format="%.4g",
    )
    st.caption(
        "Converts the file's own units to **micrometers** -- the same unit "
        "the wavelength fields below use (typing 0.5 there means 0.5 um, "
        "not 0.5 m). Match your file's raw units to micrometers, not SI "
        "meters: nm -> `1e-3`, angstrom -> `1e-4`, mm -> `1e3`. Using "
        "nm-to-*meters* (`1e-9`) instead of nm-to-*micrometers* (`1e-3`) "
        "under-scales radii by another 1e-6 and crashes MSTM."
    )
    gap_factor = st.number_input(
        "Gap factor (stretches positions only)", value=1.0, min_value=1.0, step=0.1,
    )

if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    try:
        positions = load_positions(tmp_path, scale=geom_scale, gap_factor=gap_factor)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load {uploaded.name}: {exc}")
        st.stop()
else:
    st.caption("No file uploaded -- using a default two-sphere cluster.")
    positions = np.array([[-1.5, 0.0, 0.0, 1.0], [1.5, 0.0, 0.0, 1.0]]) * geom_scale
    positions[:, :3] *= gap_factor

st.caption(f"{positions.shape[0]} spheres loaded.")

st.subheader("Material")
material_mode = st.radio(
    "Source", ["Fixed refractive index", "refidxdb database"], horizontal=True,
)
if material_mode == "Fixed refractive index":
    mc1, mc2 = st.columns(2)
    n_re = mc1.number_input("n (real)", value=1.5, step=0.1)
    n_im = mc2.number_input("n (imag)", value=0.01, step=0.01, format="%.4f")
    material = MaterialSpec(refractive_index=(n_re, n_im))
else:
    db_choice = st.selectbox("Database", list(DATABASES.keys()), format_func=str.upper)
    try:
        catalog_entries = _load_catalog(db_choice)
    except FileNotFoundError as exc:
        st.error(f"{exc}\n\nRun `refidxdb db --download {db_choice}` first.")
        catalog_entries = []

    if catalog_entries:
        entry = st.selectbox(
            f"Material ({len(catalog_entries)} available -- type to search)",
            catalog_entries,
            format_func=lambda e: _strip_html(e.label),
        )
        material = MaterialSpec(refidxdb_source=db_choice, refidxdb_catalog_path=entry.path)
        st.caption(
            "Some entries only provide n or only k (common for materials that "
            "are transparent, i.e. non-absorbing, over their measured range) -- "
            "if results come out NaN, pick a dataset that covers both."
        )
    else:
        material = None

st.subheader("Wavelength range and solver settings")
w1, w2, w3, w4 = st.columns(4)
wl_start = w1.number_input("Wavelength start (um)", value=0.4, min_value=0.01, step=0.1)
wl_stop = w1.number_input("Wavelength stop (um)", value=1.0, min_value=0.01, step=0.1)
wl_num = w1.number_input("Steps", value=5, min_value=1, max_value=50, step=1)

n_theta = w2.slider("N_theta", 5, 361, 91, step=2)
n_phi = w2.slider("N_phi", 1, 32, 1)

tolerance = w3.number_input("Tolerance", value=1e-4, format="%.1e")
max_iterations = w3.number_input("Max iterations", value=500, min_value=1, step=100)
mstm_mie_eps = w3.number_input(
    "MSTM Mie eps", value=1e-10, format="%.1e",
    help=(
        "MSTM-only: per-sphere Mie coefficient convergence tolerance. "
        "Default (1e-10) is tighter than pymstm's own library default "
        "(1e-6) -- needed for touching/near-touching spheres deep in the "
        "Rayleigh regime, where looser values under-truncate near-field "
        "coupling badly enough to flip Csca's sign. Ignored by FaSTMM2."
    ),
)
mstm_translation_eps = w3.number_input(
    "MSTM translation eps", value=1e-8, format="%.1e",
    help=(
        "MSTM-only: translation-addition-theorem convergence tolerance "
        "(near-field coupling accuracy between spheres, distinct from "
        "MSTM Mie eps's per-sphere truncation). Default (1e-8) is "
        "tighter than pymstm's own library default (1e-5), same reason "
        "as MSTM Mie eps. Ignored by FaSTMM2."
    ),
)

formulation = w4.selectbox(
    "FaSTMM2 formulation", options=[0, 1, 2],
    format_func=lambda v: {0: "STMM", 1: "FaSTMM", 2: "FaSTMM2"}[v], index=2,
)
mlfmm_accuracy = w4.slider("FaSTMM2 MLFMM accuracy (digits)", 1, 6, 2)
omp_num_threads = w4.number_input(
    "fastmm2-cli OMP_NUM_THREADS (0 = default)", value=0, min_value=0, step=1,
    help=(
        "FaSTMM2 is built with OpenMP. Only affects fastmm2-cli (a separate "
        "process); fastmm2-python runs in this process and isn't repinned. "
        "0 leaves OMP_NUM_THREADS unset, i.e. OpenMP's own default (usually "
        "all visible cores)."
    ),
)

if material is not None and st.checkbox(
    "Preview refractive index n(lambda), k(lambda) for the selected material"
):
    with st.expander("Refractive index preview", expanded=True):
        preview_wl = np.linspace(wl_start, wl_stop, max(int(wl_num), 100))
        try:
            nk = material.refractive_index_at(preview_wl)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to evaluate refractive index: {exc}")
        else:
            fig_nk = go.Figure()
            fig_nk.add_trace(go.Scatter(
                x=preview_wl, y=nk.real, name="n", mode="lines", line=dict(color="#1f77b4"),
            ))
            fig_nk.add_trace(go.Scatter(
                x=preview_wl, y=nk.imag, name="k", mode="lines", line=dict(color="#d62728"),
                yaxis="y2",
            ))
            fig_nk.update_layout(
                xaxis_title="Wavelength (um)",
                yaxis=dict(title="n", color="#1f77b4"),
                yaxis2=dict(title="k", overlaying="y", side="right", color="#d62728"),
                height=350,
            )
            st.plotly_chart(fig_nk, width="stretch")
            if np.isnan(nk).any():
                st.warning(
                    "Some points are NaN -- either this dataset only covers n or "
                    "only k, or the wavelength range extends outside what it "
                    "measures. A benchmark run will fail at those wavelengths."
                )

with st.expander("Inspect raw inputs (wavelengths + scaled positions)"):
    st.caption(
        "Exactly what gets handed to the adapters -- the wavelength array "
        "from the range/steps above, and the position file's columns "
        "after `Scale`/`Gap factor` (above, in the Cluster section) have "
        "already been applied. Use this to sanity-check sizes/units "
        "before running."
    )
    wl_array = np.linspace(wl_start, wl_stop, int(wl_num))
    st.write(f"**Wavelengths (um)**, {len(wl_array)} steps:")
    st.dataframe(
        pd.DataFrame({"wavelength_um": wl_array}), width="stretch", hide_index=True,
        column_config={"wavelength_um": st.column_config.NumberColumn(format="%.6g")},
    )
    st.write(
        f"**Scaled positions** ({positions.shape[0]} spheres, x/y/z/radius, "
        f"same units as the wavelengths above):"
    )
    st.caption(
        f"radius: min={positions[:, 3].min():.6g}, max={positions[:, 3].max():.6g}  |  "
        f"bounding box extent: "
        f"x=[{positions[:, 0].min():.6g}, {positions[:, 0].max():.6g}], "
        f"y=[{positions[:, 1].min():.6g}, {positions[:, 1].max():.6g}], "
        f"z=[{positions[:, 2].min():.6g}, {positions[:, 2].max():.6g}]"
    )
    st.dataframe(
        pd.DataFrame(positions, columns=["x", "y", "z", "radius"]),
        width="stretch", hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.6g") for c in ["x", "y", "z", "radius"]
        },
    )

st.subheader("Adapters to run")
adapter_instances = [
    Fastmm2CliAdapter(omp_num_threads=int(omp_num_threads) or None)
    if cls is Fastmm2CliAdapter else cls()
    for cls in ALL_ADAPTERS
]
adapter_cols = st.columns(len(adapter_instances))
selected_adapters = []
for col, adapter in zip(adapter_cols, adapter_instances):
    available = adapter.is_available()
    checked = col.checkbox(
        adapter.name, value=available, disabled=not available,
        help=None if available else "Not available (binary not on PATH or package not built)",
    )
    if checked:
        selected_adapters.append(adapter)

run_clicked = st.button("Run Benchmark", type="primary", disabled=(material is None or not selected_adapters))
st.divider()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if run_clicked:
    sweep = SweepRequest(
        coords=[tuple(p) for p in positions[:, :3]],
        radii=list(positions[:, 3]),
        material=material,
        wavelengths_um=list(np.linspace(wl_start, wl_stop, int(wl_num))),
        n_theta=int(n_theta), n_phi=int(n_phi),
        tolerance=float(tolerance), max_iterations=int(max_iterations),
        mstm_mie_eps=float(mstm_mie_eps), mstm_translation_eps=float(mstm_translation_eps),
        formulation=int(formulation), mlfmm_accuracy=int(mlfmm_accuracy),
    )

    progress = st.progress(0.0, text="Running benchmark...")

    def _on_progress(frac, wl_um, adapter_name):
        progress.progress(frac, text=f"{adapter_name}: wavelength {wl_um:.3g} um ({frac:.0%})")

    try:
        results = run_sweep(sweep, selected_adapters, progress_callback=_on_progress)
        st.session_state["tbench_results"] = {
            "results": results, "wavelengths_um": sweep.wavelengths_um,
        }
    except Exception as exc:  # noqa: BLE001
        st.error(f"Benchmark failed: {exc}")
    progress.empty()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

if "tbench_results" in st.session_state:
    state = st.session_state["tbench_results"]
    results = state["results"]
    wl_um = np.array(state["wavelengths_um"])

    st.subheader("Accuracy: cross sections vs. wavelength")
    acc_cols = st.columns(3)
    for col, (key, label) in zip(acc_cols, _QUANTITY_LABELS):
        fig = go.Figure()
        for adapter_name, adapter_results in results.items():
            tool = "mstm" if adapter_name.startswith("mstm") else "fastmm2"
            color = _TOOL_COLORS[tool]
            is_cli = adapter_name.endswith("cli")
            y = [getattr(r, key) if r is not None else None for r in adapter_results]
            fig.add_trace(go.Scatter(
                x=wl_um, y=y, name=adapter_name,
                mode="markers" if is_cli else "lines+markers",
                line=None if is_cli else dict(color=color),
                marker=dict(color=color, symbol="x" if is_cli else "circle"),
            ))
        fig.update_layout(
            title=label, xaxis_title="Wavelength (um)", yaxis_title=label, height=380,
        )
        col.plotly_chart(fig, width="stretch")

    err_setting_cols = st.columns(2)
    gt_tool = err_setting_cols[0].radio(
        "Ground truth tool", ["MSTM", "FaSTMM2"], horizontal=True,
    )
    error_mode = err_setting_cols[1].radio(
        "Error type", ["Relative: (val - gt) / gt", "Ratio: val / gt"], horizontal=True,
    )
    # Prefer the *-python adapter (direct f2py call, no file-round-trip
    # precision loss) over its *-cli counterpart if both ran.
    gt_candidates = ("mstm-python", "mstm-cli") if gt_tool == "MSTM" else ("fastmm2-python", "fastmm2-cli")
    gt_name = next((n for n in gt_candidates if n in results), None)
    if gt_name is None:
        st.info(f"Select a {gt_tool} adapter to see the error plot below.")
    else:
        is_relative = error_mode.startswith("Relative")
        y_title = "Error (%)" if is_relative else "Ratio (val / gt)"
        hline_y = 0 if is_relative else 1
        st.subheader(f"Error vs. {gt_tool} ground truth ({gt_name}): {error_mode.split(': ', 1)[1]}")
        gt_results = results[gt_name]
        err_cols = st.columns(3)
        for col, (key, label) in zip(err_cols, _QUANTITY_LABELS):
            fig = go.Figure()
            for adapter_name, adapter_results in results.items():
                if adapter_name == gt_name:
                    continue
                tool = "mstm" if adapter_name.startswith("mstm") else "fastmm2"
                color = _TOOL_COLORS[tool]
                is_cli = adapter_name.endswith("cli")
                y = []
                for r, gt in zip(adapter_results, gt_results):
                    if r is None or gt is None or getattr(gt, key) == 0:
                        y.append(None)
                    elif is_relative:
                        y.append(100 * (getattr(r, key) - getattr(gt, key)) / getattr(gt, key))
                    else:
                        y.append(getattr(r, key) / getattr(gt, key))
                fig.add_trace(go.Scatter(
                    x=wl_um, y=y, name=adapter_name,
                    mode="markers" if is_cli else "lines+markers",
                    line=None if is_cli else dict(color=color),
                    marker=dict(color=color, symbol="x" if is_cli else "circle"),
                ))
            fig.add_hline(y=hline_y, line=dict(color="gray", dash="dot"))
            fig.update_layout(
                title=label, xaxis_title="Wavelength (um)", yaxis_title=y_title, height=380,
            )
            col.plotly_chart(fig, width="stretch")

    st.subheader("Speed: wall-clock time vs. wavelength")
    fig_time = go.Figure()
    for adapter_name, adapter_results in results.items():
        tool = "mstm" if adapter_name.startswith("mstm") else "fastmm2"
        color = _TOOL_COLORS[tool]
        is_cli = adapter_name.endswith("cli")
        y = [r.wall_time_seconds if r is not None else None for r in adapter_results]
        fig_time.add_trace(go.Scatter(
            x=wl_um, y=y, name=adapter_name,
            mode="markers" if is_cli else "lines+markers",
            line=None if is_cli else dict(color=color),
            marker=dict(color=color, symbol="x" if is_cli else "circle"),
        ))
    fig_time.update_layout(
        xaxis_title="Wavelength (um)", yaxis_title="Wall time (s)", yaxis_type="log",
    )
    st.plotly_chart(fig_time, width="stretch")

    with st.expander("Results table and accuracy", expanded=True):
        # Cext can legitimately span many orders of magnitude depending on
        # cluster scale/wavelength (e.g. sub-picometer^2 for a cluster
        # scaled to tens-of-nm particles) -- round(x, 6) silently displayed
        # anything below 1e-6 as a bare "0", which is exactly what a user
        # ran into. Values stay full-precision floats; only the *display*
        # goes through NumberColumn's scientific-notation format, so the
        # table stays sortable/exportable as real numbers.
        rows = []
        cext_cols = []
        time_cols = []
        for i, wl in enumerate(wl_um):
            row: dict[str, object] = {"wavelength_um": float(wl)}
            c_ext_values = []
            for adapter_name, adapter_results in results.items():
                r = adapter_results[i]
                cext_col, time_col = f"{adapter_name} Cext", f"{adapter_name} time (s)"
                cext_cols.append(cext_col)
                time_cols.append(time_col)
                if r is None:
                    row[cext_col] = None
                    row[time_col] = None
                else:
                    row[cext_col] = r.c_ext
                    row[time_col] = r.wall_time_seconds
                    c_ext_values.append(r.c_ext)
            if len(c_ext_values) >= 2:
                row["max Cext spread (%)"] = (
                    (max(c_ext_values) - min(c_ext_values)) / max(c_ext_values) * 100
                )
            rows.append(row)

        column_config = {"wavelength_um": st.column_config.NumberColumn(format="%.4g")}
        for col in set(cext_cols):
            column_config[col] = st.column_config.NumberColumn(format="scientific")
        for col in set(time_cols):
            column_config[col] = st.column_config.NumberColumn(format="%.4f")
        column_config["max Cext spread (%)"] = st.column_config.NumberColumn(format="%.3f")

        st.dataframe(
            pd.DataFrame(rows), width="stretch", hide_index=True, column_config=column_config,
        )
else:
    st.info("Configure a cluster, material, and wavelength range above, then click **Run Benchmark**.")
