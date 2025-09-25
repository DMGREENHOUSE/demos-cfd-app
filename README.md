
# demos-cfd-app

A tiny Streamlit app to **generate synthetic CFD-like datasets** and download them as CSV.  
It supports **one or multiple input variables** (you choose the names and bounds) and produces a **velocity profile** at ~N spatial locations.

> This app **does not** fit an emulator; it only generates data to demonstrate how you would later train a surrogate (e.g., a GP) offline.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Features

- Configure **1–5 input variables** (names + bounds)
- Choose which variables are **variable vs fixed**
- **Random** or **structured-grid** sampling
- **Preview** inputs & outputs, **plot**, and **download** a combined CSV with a custom filename

Columns in the downloaded CSV are:
```
<input_1>, <input_2>, ..., <input_K>, u_0, u_1, ..., u_{M-1}
```

## License

MIT


## CLI

You can generate the same dataset without the UI:

```bash
python -m scripts.generate_dataset --inputs "Re,PR" --mins "1e3,0" --maxs "5e5,1" --n 500 --sampling random   --gap-var Re --gap "2e4,6e4" --M 10 --plateaus-on --out data/train.csv
```

Or evaluate pre-defined input points:
```bash
python -m scripts.generate_dataset --inputs "Re,PR" --mins "1e3,0" --maxs "5e5,1"   --inputs-csv my_inputs.csv --out data/from_inputs.csv
```


### Defaults & Theme

- Noise defaults to **0** (noise_low = noise_mid = noise_high = 0).
- Default number of points in auto-sampling: **20**.
- By default, **all inputs are variable** (you can adjust in the expander).
- Sidebar shows only **number of inputs** and **number of outputs**.
- Theme is set via `.streamlit/config.toml` with brand colors.
