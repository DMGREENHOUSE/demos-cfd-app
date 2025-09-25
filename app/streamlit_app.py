
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Callable
import math

# Ensure src package is importable for local runs
import os, sys
ROOT = os.path.dirname(__file__)
REPO = os.path.dirname(ROOT)
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from demos_cfd_app.synthetic_model import compute_velocity_profiles

# -----------------------------
# Branding palette
# -----------------------------
KEY_LIME   = "#EBF38B"
INDIGO     = "#16425B"
INDIGO_50  = "#8AA0AD"
KEPPEL     = "#16D5C2"
KEPPEL_50  = "#8AEAE1"
BLACK      = "#000000"
GREY_80    = "#333333"

# Try Altair for branded scatter; fall back gracefully
try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

# =========================================
# Shared UI helpers (single-module version)
# =========================================
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

@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def compatible_append(base_df: pd.DataFrame, new_df: pd.DataFrame):
    base_cols = list(base_df.columns)
    new_cols = list(new_df.columns)
    if set(base_cols) == set(new_cols):
        return True, new_df[base_cols], None
    missing = [c for c in base_cols if c not in new_cols]
    extra = [c for c in new_cols if c not in base_cols]
    return False, None, f"Column mismatch. Missing: {missing}; Extra: {extra}"

def variable_fixed_controls(label: str, ranges: Dict[str, Tuple[float, float]], defaults: Dict[str, float]):
    st.subheader(f"{label} • Variable/Fixed Controls")
    n = st.number_input(
        "Number of points (per sweep / random batch)",
        min_value=1, max_value=200_000, value=20, step=10, key=f"{label}_vf_n"
    )

    var_flags: Dict[str, bool] = {}
    var_ranges: Dict[str, Tuple[float, float]] = {}
    fixed_vals: Dict[str, float] = {}

    with st.expander("Select variables to vary and set ranges / fixed values", expanded=False):
        for i, (k, (a, b)) in enumerate(ranges.items()):
            cols = st.columns([1.4, 1, 1, 1])
            var_flags[k] = cols[0].checkbox(
                f"{k} is variable",
                value=True if f"{label}_{k}_is_var" not in st.session_state else None,
                key=f"{label}_{k}_is_var"
            )
            if var_flags[k]:
                new_a = cols[1].number_input(f"{k} min", value=float(a), key=f"{label}_{k}_a_vf")
                new_b = cols[2].number_input(f"{k} max", value=float(b), key=f"{label}_{k}_b_vf")
                if new_b < new_a:
                    st.warning(f"Adjusted {k} max to be ≥ min")
                    new_b = new_a
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
    df = pd.DataFrame(init)
    return st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"{label}_custom_editor")

def upload_inputs(label: str, ranges: Dict[str, Tuple[float, float]]):
    st.subheader(f"{label} • Upload Inputs")
    st.caption(f"Expected columns: {list(ranges.keys())}")
    file = st.file_uploader(f"Upload inputs CSV for {label}", type=["csv"], key=f"{label}_inputs_uploader")
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        missing = [c for c in ranges.keys() if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

def append_controls(label: str):
    append = st.checkbox("Append new data to an uploaded base dataset", key=f"{label}_append_toggle")
    base_df = None
    if append:
        st.caption("Upload base dataset (must share the same columns as the generated combined dataset).")
        up = st.file_uploader(f"Upload base dataset CSV for {label}", type=["csv"], key=f"{label}_append_uploader")
        if up is not None:
            try:
                base_df = pd.read_csv(up)
            except Exception as e:
                st.error(f"Failed to read base dataset: {e}")
                base_df = None
    return append, base_df

def preview_and_download(key_prefix: str, default_prefix: str):
    X = st.session_state.get(f"{key_prefix}_inputs")
    y = st.session_state.get(f"{key_prefix}_outputs")
    combined = st.session_state.get(f"{key_prefix}_combined")
    appended = st.session_state.get(f"{key_prefix}_appended")

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
        collapsed = st.session_state.get(f"{key_prefix}_collapsed", False)
        with st.expander("Preview, Plot & Download", expanded=not collapsed):
            st.subheader("Preview: Inputs")
            st.dataframe(X, use_container_width=True)
            st.subheader("Preview: Outputs")
            st.dataframe(y, use_container_width=True)

            st.subheader("Plot (scatter)")
            colx, coly = st.columns(2)
            with colx:
                x_var = st.selectbox("X axis (input variable)", list(X.columns), key=f"{key_prefix}_plot_x")
            with coly:
                y_var = st.selectbox("Y axis (output variable)", list(y.columns), key=f"{key_prefix}_plot_y")

            plot_df = pd.DataFrame({x_var: X[x_var].to_numpy(), y_var: y[y_var].to_numpy()})

            if alt is not None:
                chart = (
                    alt.Chart(plot_df)
                    .mark_circle(size=64, color=INDIGO)
                    .encode(
                        x=alt.X(x_var, title=x_var),
                        y=alt.Y(y_var, title=y_var),
                        tooltip=[x_var, y_var],
                    )
                    .interactive()
                    .configure_axis(labelColor=BLACK, titleColor=BLACK, gridOpacity=0.15)
                    .configure_view(strokeOpacity=0)
                    .configure_title(color=BLACK)
                    .configure(background='white')
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Altair unavailable — using Streamlit fallback plot.")
                st.scatter_chart(plot_df, x=x_var, y=y_var, use_container_width=True)

            st.markdown("### Download")
            target_df = appended if isinstance(appended, pd.DataFrame) else combined
            default_name = (
                f"{default_prefix}_appended.csv"
                if isinstance(appended, pd.DataFrame)
                else f"{default_prefix}_dataset.csv"
            )
            st.text_input(
                "Dataset CSV filename. Press Enter to set.",
                value=default_name,
                key=f"{key_prefix}_dl_name",
            )
            current_name = st.session_state.get(f"{key_prefix}_dl_name", default_name)
            clicked = st.download_button(
                label="Download dataset CSV",
                data=to_csv_bytes(target_df),
                file_name=current_name,
                mime="text/csv",
                key=f"{key_prefix}_dl_btn",
            )
            if clicked:
                st.session_state[f"{key_prefix}_collapsed"] = True

        if collapsed:
            st.info("Preview collapsed after download. Use the button below to reopen.")
            if st.button("Reopen Preview", key=f"{key_prefix}_reopen"):
                st.session_state[f"{key_prefix}_collapsed"] = False

def _make_structured_grid(ranges: Dict[str, Tuple[float, float]], steps: Dict[str, int]) -> pd.DataFrame:
    if not ranges:
        return pd.DataFrame({"_dummy": [0]})
    axes = []
    keys = list(ranges.keys())
    for k in keys:
        a, b = ranges[k]
        n = max(1, int(steps.get(k, 10)))
        arr = np.linspace(a, b, n) if n > 1 else np.array([(a + b) / 2.0])
        axes.append(arr)
    grids = np.meshgrid(*axes, indexing="xy")
    flat = [g.reshape(-1) for g in grids]
    return pd.DataFrame({k: v for k, v in zip(keys, flat)})

def run_tab(
    *,
    label: str,
    key_prefix: str,
    ranges: Dict[str, Tuple[float, float]],
    compute_fn: Callable,
    rng: np.random.Generator,
    default_prefix: str,
    sampling_default: str = "Random (uniform)",
):
    apply_branding()

    defaults = {k: (a + b) / 2 for k, (a, b) in ranges.items()}

    st.divider()
    st.markdown("### Input Source")
    input_mode = st.radio(
        "Choose how to provide inputs",
        ["Auto sampling", "Custom points", "Upload inputs"],
        key=f"{label}_input_mode",
        horizontal=True,
    )

    append, base_df = append_controls(label)

    X = None
    if input_mode == "Auto sampling":
        n, var_flags, var_ranges, fixed_vals = variable_fixed_controls(label, ranges, defaults)

        sampling_type = st.selectbox(
            "Sampling type",
            ["Random (uniform)", "Structured grid"],
            index=0 if sampling_default == "Random (uniform)" else 1,
            key=f"{label}_sampling_type"
        )

        variable_keys = [k for k, flag in var_flags.items() if flag]

        if len(variable_keys) == 0:
            st.info("No variables selected — all inputs are fixed. Generates a single row.")
            X = {k: np.array([fixed_vals[k]]) for k in ranges.keys()}
            X = pd.DataFrame(X)

        elif len(variable_keys) == 1:
            sweep_key = variable_keys[0]
            a, b = var_ranges[sweep_key]

            if sampling_type == "Random (uniform)":
                x = rng.uniform(a, b, size=n)
            else:  # structured
                x = np.linspace(a, b, n)

            data = {}
            for k in ranges.keys():
                if k == sweep_key:
                    data[k] = x
                else:
                    data[k] = np.full(n, fixed_vals[k])
            X = pd.DataFrame(data)

        else:
            if sampling_type == "Random (uniform)":
                data = {}
                for k in ranges.keys():
                    if var_flags[k]:
                        a, b = var_ranges[k]
                        data[k] = rng.uniform(a, b, size=n)
                    else:
                        data[k] = np.full(n, fixed_vals[k])
                X = pd.DataFrame(data)

            else:  # Structured grid with automatic steps
                d = len(variable_keys)
                steps_float = n ** (1 / d)
                steps_int = max(2, int(round(steps_float)))
                total_pts = steps_int ** d
                st.caption(f"Structured grid with ~{n:,} target points → "
                           f"{steps_int} steps per variable → {total_pts:,} total points")

                if total_pts > 100_000:
                    st.error("Structured grid too large (>100,000 points). Reduce requested points.")
                    X = None
                else:
                    grid_ranges = {k: var_ranges[k] for k in variable_keys}
                    grid_steps  = {k: steps_int for k in variable_keys}
                    grid_df = _make_structured_grid(grid_ranges, grid_steps)
                    for k in ranges.keys():
                        if not var_flags[k]:
                            grid_df[k] = fixed_vals[k]
                    X = grid_df[[*ranges.keys()]]

        if st.button(f"Generate {label} Data", key=f"{label}_go_vf"):
            pass

    elif input_mode == "Custom points":
        edited = custom_points_editor(label, ranges, defaults)
        if st.button(f"Generate {label} Data from Custom Points", key=f"{label}_go_custom"):
            X = edited.copy()

    else:  # Upload
        up = upload_inputs(label, ranges)
        if st.button(f"Generate {label} Data from Uploaded Inputs", key=f"{label}_go_upload"):
            if up is None:
                st.error("Please upload a valid inputs CSV with the expected columns.")
            else:
                X = up.copy()

    # Compute & assemble with **defaults** (no model/gap/noise controls in UI)
    if isinstance(X, pd.DataFrame):
        # Derive simple defaults from ranges
        primary_var = list(ranges.keys())[0]
        pr_a, pr_b = ranges[primary_var]
        params = dict(
            M=st.session_state.get("M", 10),
            primary_var=primary_var,
            plateaus_on=True,
            plateau_low=(pr_a, pr_a + 0.2*(pr_b - pr_a)),
            plateau_high=(pr_a + 0.7*(pr_b - pr_a), pr_b),
            transition_center=0.5*(pr_a + pr_b),
            transition_width=max(1e-6, 0.1*(pr_b - pr_a)),
            spatial_amp1=0.15,
            spatial_amp2=0.10,
            interaction_strength=0.10 if len(ranges) > 1 else 0.0,
            noise_low=0.0,
            noise_mid=0.0,
            noise_high=0.0,
        )

        y = compute_velocity_profiles(X, rng=rng, **params)
        combined = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

        st.session_state[f"{key_prefix}_inputs"] = X
        st.session_state[f"{key_prefix}_outputs"] = y
        st.session_state[f"{key_prefix}_combined"] = combined
        st.session_state.pop(f"{key_prefix}_appended", None)

        if append and isinstance(base_df, pd.DataFrame):
            ok, reordered_new, msg = compatible_append(base_df, combined)
            if ok:
                st.session_state[f"{key_prefix}_appended"] = pd.concat([base_df, reordered_new], ignore_index=True)
                st.success("Appended to base dataset.")
            else:
                st.warning(f"Append skipped: {msg}")
        elif append and base_df is None:
            st.warning("Append requested but no base dataset uploaded.")

        st.success(f"Generated {label} dataset.")

    preview_and_download(key_prefix, default_prefix)

# =========================================
# Page (simplified controls)
# =========================================
st.set_page_config(page_title="demos-cfd-app • CSV-only", layout="wide")
st.title("🌀 demos-cfd-app — CFD-like Dataset Generator (CSV only)")
st.caption("Configure 1–5 inputs, generate velocity profiles with sensible defaults, and download a CSV.")


with st.sidebar:
    st.header("Inputs configuration")
    n_inputs = st.slider("Number of input variables", 1, 5, 5, 1)
    # Use default names and ranges internally (not editable here)
    default_names = ["Re", "PR", "Tin", "Rough", "Angle"]
    ranges: Dict[str, Tuple[float, float]] = {}
    for i in range(n_inputs):
        name = default_names[i]
        if i == 0:
            a, b = 1e3, 5e5   # Reynolds-like
        else:
            a, b = 0.0, 1.0    # Normalized secondary inputs
        ranges[name] = (float(a), float(b))

    st.markdown("---")
    st.header("Outputs")
    M = st.slider("Number of velocity locations", 5, 50, 10, 1)
    st.session_state["M"] = M


rng = np.random.default_rng(42)

run_tab(
    label="Velocity Profiles",
    key_prefix="vel",
    ranges=ranges,
    compute_fn=compute_velocity_profiles,  # used via params above
    rng=rng,
    default_prefix="cfd_velocity",
    sampling_default="Random (uniform)",
)
