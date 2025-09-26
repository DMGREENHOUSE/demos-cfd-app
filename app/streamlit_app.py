
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import os, sys

ROOT = os.path.dirname(__file__)
REPO = os.path.dirname(ROOT)
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from demos_cfd_app.synthetic_model import compute_velocity_profiles

KEY_LIME   = "#EBF38B"
INDIGO     = "#16425B"
INDIGO_50  = "#8AA0AD"
KEPPEL     = "#16D5C2"
KEPPEL_50  = "#8AEAE1"
BLACK      = "#000000"
GREY_80    = "#333333"

DEFAULT_VARY_INIT = ["wall_roughness", "turbulence_intensity", "inlet_temperature"]

try:
    import altair as alt
except Exception:
    alt = None

def apply_branding():
    if st.session_state.get("_branding_injected"):
        return
    BRAND_CSS = f"""
    <style>
      :root {{
        --brand-primary: {INDIGO};
        --brand-primary-50: {INDIGO_50};
        --brand-accent: {KEPPEL};
        --brand-accent-50: {KEPPEL_50};
        --brand-accent-2: {KEY_LIME};
        --brand-black: {BLACK};
        --brand-grey-80: {GREY_80};
      }}
      h1, h2, h3 {{ color: var(--brand-black); }}
      div.stButton > button, div.stDownloadButton > button {{
        background-color: var(--brand-primary);
        color: white;
        border: 0;
        border-radius: 12px;
      }}
      div.stButton > button:hover, div.stDownloadButton > button:hover {{
        background-color: var(--brand-primary-50);
      }}
      div.streamlit-expanderHeader {{
        background: linear-gradient(90deg, var(--brand-accent-50) 0%, var(--brand-accent-2) 100%);
        color: var(--brand-black);
        border-radius: 8px;
      }}
      label {{ color: var(--brand-grey-80); }}
      [data-testid="stTable"] thead th {{
        background-color: var(--brand-accent-50);
        color: var(--brand-black);
      }}
    </style>
    """
    st.markdown(BRAND_CSS, unsafe_allow_html=True)
    st.session_state["_branding_injected"] = True
UNITS = {
    "wall_roughness": "mm",
    "turbulence_intensity": "%",
    "inlet_temperature": "K",
    "inlet_velocity": "m/s",
    "outlet_pressure": "Pa",
}
for i in range(100): UNITS[f"u_{i}"] = "Norm."

@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def variable_fixed_controls(
    label: str,
    ranges: Dict[str, Tuple[float, float]],
    defaults: Dict[str, float],
    default_vary: list[str] | None = None
):
    st.subheader(f"{label} • Variable/Fixed Controls")
    n = st.number_input(
        "Number of points (per sweep / random batch)",
        min_value=1, max_value=10_000, value=10, step=10,
        key=f"{label}_vf_n"
    )

    var_flags: Dict[str, bool] = {}
    var_ranges: Dict[str, Tuple[float, float]] = {}
    fixed_vals: Dict[str, float] = {}

    default_vary = set(default_vary or [])

    with st.expander("Select variables to vary and set ranges / fixed values", expanded=False):
        for i, (k, (a, b)) in enumerate(ranges.items()):
            cols = st.columns([1.6, 1, 1, 1.2])

            # If it's the first time, use our default; otherwise let Streamlit keep the user’s choice
            checkbox_key = f"{label}_{k}_is_var"
            default_is_var = (k in default_vary)
            var_flags[k] = cols[0].checkbox(
                f"{k} is variable",
                value=default_is_var if checkbox_key not in st.session_state else None,
                key=checkbox_key,
                disabled=False  # everyone can vary
            )

            if var_flags[k]:
                new_a = cols[1].number_input(f"{k} min", value=float(a), key=f"{label}_{k}_a_vf")
                new_b = cols[2].number_input(f"{k} max", value=float(b), key=f"{label}_{k}_b_vf")
                if new_b < new_a:
                    st.warning(f"Adjusted {k} max to be ≥ min"); new_b = new_a
                var_ranges[k] = (new_a, new_b)
                cols[3].markdown("&nbsp;")
            else:
                c = defaults.get(k, (a + b) / 2)
                fixed_vals[k] = cols[3].number_input(f"{k} (fixed)", value=float(c), key=f"{label}_{k}_fixed_vf")

    return int(n), var_flags, var_ranges, fixed_vals


def custom_points_editor(label: str, ranges: Dict[str, Tuple[float, float]], defaults: Dict[str, float]) -> pd.DataFrame:
    st.subheader(f"{label} • Custom Points")
    st.caption("Add/remove rows to choose exact input points where the model will be evaluated.")
    init = {k: [defaults.get(k, (a + b) / 2)] for k, (a, b) in ranges.items()}
    return st.data_editor(pd.DataFrame(init), num_rows="dynamic", use_container_width=True, key=f"{label}_custom_editor")

def upload_inputs(label: str, ranges: Dict[str, Tuple[float, float]]):
    st.subheader(f"{label} • Upload Inputs")
    st.caption(f"Expected columns: {list(ranges.keys())}")
    file = st.file_uploader(f"Upload inputs CSV for {label}", type=["csv"], key=f"{label}_inputs_uploader")
    if file is None: return None
    try:
        df = pd.read_csv(file)
        missing = [c for c in ranges.keys() if c not in df.columns]
        if missing: st.error(f"Missing columns: {missing}"); return None
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}"); return None

def preview_and_download(key_prefix: str, default_prefix: str):
    X = st.session_state.get(f"{key_prefix}_inputs")
    y = st.session_state.get(f"{key_prefix}_outputs")
    combined = st.session_state.get(f"{key_prefix}_combined")
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
        with st.expander("Preview, Plot & Download", expanded=True):
            st.subheader("Preview: Inputs"); st.dataframe(X, use_container_width=True)
            st.subheader("Preview: Outputs"); st.dataframe(y, use_container_width=True)
            st.subheader("Plot")
            colx, coly = st.columns(2)
            with colx: x_var = st.selectbox("X axis (input variable)", list(X.columns), key=f"{key_prefix}_plot_x")
            with coly: y_var = st.selectbox("Y axis (output variable)", list(y.columns), key=f"{key_prefix}_plot_y")
            plot_df = pd.DataFrame({x_var: X[x_var].to_numpy(), y_var: y[y_var].to_numpy()})
            try:
                chart = (
                    alt.Chart(plot_df)
                    .mark_circle(size=64, color=INDIGO)
                    .encode(
                        x=alt.X(x_var, title=f"{x_var} [{UNITS.get(x_var, '')}]"),
                        y=alt.Y(y_var, title=f"{y_var} [{UNITS.get(y_var, '')}]"),
                        tooltip=[x_var, y_var],
                    )
                    .interactive()
                    .properties(width=400, height=400)
                )
                st.altair_chart(chart, use_container_width=False)
            except Exception:
                st.scatter_chart(plot_df, x=x_var, y=y_var, use_container_width=True)
            st.markdown("### Download")
            default_name = f"{default_prefix}.csv"
            st.text_input("Dataset CSV filename. Press Enter to set.", value=default_name, key=f"{key_prefix}_dl_name")
            current_name = st.session_state.get(f"{key_prefix}_dl_name", default_name)
            st.download_button("Download dataset CSV", data=to_csv_bytes(combined), file_name=current_name, mime="text/csv", key=f"{key_prefix}_dl_btn")

def run_tab(*, label: str, key_prefix: str, ranges: Dict[str, Tuple[float, float]], rng: np.random.Generator, default_prefix: str):
    apply_branding()
    defaults = {k: (a + b) / 2 for k, (a, b) in ranges.items()}
    st.divider(); st.markdown("### Input Source")
    input_mode = st.radio("Choose how to provide inputs", ["Auto sampling", "Custom points", "Upload inputs"], key=f"{label}_input_mode", horizontal=True)
    X = None
    if input_mode == "Auto sampling":
        n, var_flags, var_ranges, fixed_vals = variable_fixed_controls(
            label, ranges, defaults, default_vary=DEFAULT_VARY_INIT
        )

        # n, var_flags, var_ranges, fixed_vals = variable_fixed_controls(label, ranges, defaults)
        sampling_type = st.selectbox("Sampling type", ["Random (uniform)", "Structured grid"], index=0, key=f"{label}_sampling_type")
        variable_keys = [k for k, flag in var_flags.items() if flag]
        if len(variable_keys) == 0:
            X = pd.DataFrame({k: [fixed_vals[k]] for k in ranges.keys()})
        elif len(variable_keys) == 1:
            sweep_key = variable_keys[0]; a, b = var_ranges[sweep_key]
            x = rng.uniform(a, b, size=n) if sampling_type == "Random (uniform)" else np.linspace(a, b, n)
            data = {k: (x if k == sweep_key else np.full(n, fixed_vals[k])) for k in ranges.keys()}; X = pd.DataFrame(data)
        else:
            if sampling_type == "Random (uniform)":
                data = {k: (rng.uniform(*var_ranges[k], size=n) if var_flags[k] else np.full(n, fixed_vals[k])) for k in ranges.keys()}
                X = pd.DataFrame(data)
            else:
                d = len(variable_keys); steps = max(2, int(round(n ** (1 / d)))); total = steps ** d
                if total > 100_000: st.error("Structured grid too large (>100,000 points). Reduce requested points."); X=None
                else:
                    axes = [np.linspace(*var_ranges[k], steps) for k in variable_keys]
                    grids = np.meshgrid(*axes, indexing="xy"); flat = [g.reshape(-1) for g in grids]
                    grid_df = pd.DataFrame({k: v for k, v in zip(variable_keys, flat)})
                    for k in ranges.keys():
                        if not var_flags[k]: grid_df[k] = fixed_vals[k]
                    X = grid_df[[*ranges.keys()]]
        if st.button(f"Generate {label} Data", key=f"{label}_go_vf"): pass
    elif input_mode == "Custom points":
        edited = custom_points_editor(label, ranges, defaults)
        if st.button(f"Generate {label} Data from Custom Points", key=f"{label}_go_custom"): X = edited.copy()
    else:
        up = upload_inputs(label, ranges)
        if st.button(f"Generate {label} Data from Uploaded Inputs", key=f"{label}_go_upload"):
            if up is None: st.error("Please upload a valid inputs CSV with the expected columns.")
            else: X = up.copy()
    if isinstance(X, pd.DataFrame):
        params = dict(M=st.session_state.get("M", 10), spatial_amp1=0.15, spatial_amp2=0.10, interaction_strength=0.10 if len(ranges)>1 else 0.0,
                      noise_low=0.0, noise_mid=0.0, noise_high=0.0)
        y = compute_velocity_profiles(X, rng=rng, **params); combined = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        st.session_state[f"{key_prefix}_inputs"] = X; st.session_state[f"{key_prefix}_outputs"] = y; st.session_state[f"{key_prefix}_combined"] = combined
        st.success(f"Generated {label} dataset.")
    preview_and_download(key_prefix, default_prefix)

st.set_page_config(page_title="demos-cfd-app • CSV-only", layout="wide")
st.title("CFD-like Dataset Generator")

# Image banner
img_path = os.path.join(REPO, "assets", "cfd_flow.png")
col1, col2 = st.columns([1,1])  # two equal halves
with col1:
    st.image("assets/cfd_flow.png", use_container_width=True)
with col2:
    st.empty()  # leave blank or add text
st.caption("Configure inputs, generate velocity profiles and download a CSV.")

with st.sidebar:
    st.header("Inputs configuration")
    n_inputs = st.slider("Number of input variables", 1, 5, 5, 1)
    default_names = ["wall_roughness", "turbulence_intensity", "inlet_temperature", "inlet_velocity", "outlet_pressure"]
    ranges: Dict[str, Tuple[float, float]] = {}
    for i in range(n_inputs):
        name = default_names[i]
        if name == "turbulence_intensity": a, b = 0.0, 20.0       # %
        elif name == "inlet_temperature":  a, b = 250.0, 400.0    # K
        elif name == "inlet_velocity":     a, b = 0.0, 50.0       # m/s
        elif name == "outlet_pressure":    a, b = 0.0, 1e5        # Pa
        elif name == "wall_roughness":     a, b = 0.0, 1.0        # mm
        else:                               a, b = 0.0, 1.0
        ranges[name] = (float(a), float(b))

    st.markdown("---")
    st.header("Outputs")
    M = st.slider("Number of velocity locations", 5, 50, 10, 1)
    st.session_state["M"] = M

rng = np.random.default_rng(42)
run_tab(label="Velocity Profiles", key_prefix="vel", ranges=ranges, rng=rng, default_prefix="cfd_velocity")

# Parameters & Units summary (collapsed)
with st.expander("Parameters & Units (summary)", expanded=False):
    units_df = pd.DataFrame([
        {"Parameter": "wall_roughness", "Units": "mm"},
        {"Parameter": "turbulence_intensity", "Units": "%"},
        {"Parameter": "inlet_temperature", "Units": "K"},
        {"Parameter": "inlet_velocity", "Units": "m/s"},
        {"Parameter": "outlet_pressure", "Units": "Pa"},
        {"Parameter": "u_0 .. u_{M-1}", "Units": "[Norm.]"},
    ])
    st.table(units_df)
