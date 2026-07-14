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

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tbench import ALL_ADAPTERS, MaterialSpec, SweepRequest, load_positions, run_sweep

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
        help="Use to convert the file's own units to micrometers if it isn't already.",
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
    "Source", ["Fixed refractive index", "refidxdb URL"], horizontal=True,
)
if material_mode == "Fixed refractive index":
    mc1, mc2 = st.columns(2)
    n_re = mc1.number_input("n (real)", value=1.5, step=0.1)
    n_im = mc2.number_input("n (imag)", value=0.01, step=0.01, format="%.4f")
    material = MaterialSpec(refractive_index=(n_re, n_im))
else:
    refidxdb_url = st.text_input(
        "refidxdb URL (refractiveindex.info or eodg.atm.ox.ac.uk)",
        value="https://refractiveindex.info/database/data/main/SiO2/nk/Rodriguez-de%20Marcos.yml",
    )
    material = MaterialSpec(refidxdb_url=refidxdb_url) if refidxdb_url else None

st.subheader("Wavelength range and solver settings")
w1, w2, w3, w4 = st.columns(4)
wl_start = w1.number_input("Wavelength start (um)", value=0.4, min_value=0.01, step=0.1)
wl_stop = w1.number_input("Wavelength stop (um)", value=1.0, min_value=0.01, step=0.1)
wl_num = w1.number_input("Steps", value=5, min_value=1, max_value=50, step=1)

n_theta = w2.slider("N_theta", 5, 361, 91, step=2)
n_phi = w2.slider("N_phi", 1, 32, 1)

tolerance = w3.number_input("Tolerance", value=1e-4, format="%.1e")
max_iterations = w3.number_input("Max iterations", value=500, min_value=1, step=100)

formulation = w4.selectbox(
    "FaSTMM2 formulation", options=[0, 1, 2],
    format_func=lambda v: {0: "STMM", 1: "FaSTMM", 2: "FaSTMM2"}[v], index=2,
)
mlfmm_accuracy = w4.slider("FaSTMM2 MLFMM accuracy (digits)", 1, 6, 2)

st.subheader("Adapters to run")
adapter_instances = [cls() for cls in ALL_ADAPTERS]
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
        rows = []
        for i, wl in enumerate(wl_um):
            row: dict[str, object] = {"wavelength_um": round(float(wl), 4)}
            c_ext_values = []
            for adapter_name, adapter_results in results.items():
                r = adapter_results[i]
                if r is None:
                    row[f"{adapter_name} Cext"] = None
                    row[f"{adapter_name} time (s)"] = None
                else:
                    row[f"{adapter_name} Cext"] = round(r.c_ext, 6)
                    row[f"{adapter_name} time (s)"] = round(r.wall_time_seconds, 4)
                    c_ext_values.append(r.c_ext)
            if len(c_ext_values) >= 2:
                spread = (max(c_ext_values) - min(c_ext_values)) / max(c_ext_values) * 100
                row["max Cext spread (%)"] = round(spread, 3)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
else:
    st.info("Configure a cluster, material, and wavelength range above, then click **Run Benchmark**.")
