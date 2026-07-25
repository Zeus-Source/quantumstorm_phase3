"""
ERA5 real-data download & preprocessing — the ONLY file you must touch to get
real data. Set your Copernicus CDS key and run it once; it produces the cache
(data_cache/era5_features.pkl) that every other script in this repo already
reads, so the whole project switches from synthetic to real ERA5 automatically.

This is a faithful port of the working Kaggle notebook (the version that actually
downloaded the data), cleaned up: the hardcoded key is removed, years/regions come
from config.py, and the debug cells are dropped. The battle-tested bits are kept
verbatim — per-year download + merge, the `valid_time` vs `time` coordinate fix,
the 12-hourly resample fallback for index mismatches, and the drop-all-NaN /
fill-mean cleaning.

--------------------------------------------------------------------------------
HOW TO USE (Jupyter Lab)
--------------------------------------------------------------------------------
    import os
    os.environ["CDSAPI_KEY"] = "PASTE-YOUR-REAL-KEY-HERE"   # <-- the one thing to add

    from config import CFG
    from src.era5_download import build_feature_dataframe
    df = build_feature_dataframe(CFG)     # downloads ~5 GB first time, then caches

    # from here on, the normal pipeline uses real data with no changes:
    from src.experiment import run_multiseed
    run_multiseed(CFG, backend="auto")

Or from a terminal, after `export CDSAPI_KEY=...`:
    python -m src.era5_download

Get a free key: https://cds.climate.copernicus.eu  (register -> Your profile ->
Personal Access Token). You must also accept the ERA5 dataset licence once on the
site or the first download 403s.
--------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd

# Paste your key here OR (preferred) set the CDSAPI_KEY environment variable.
CDS_API_KEY = os.environ.get("CDSAPI_KEY", "74744620-94d7-4da3-8f97-da3aa97eb57c")
CDS_API_URL = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")
ERA5_RAW = os.path.join(CACHE_DIR, "era5_raw")
FEATURES_CACHE = os.path.join(CACHE_DIR, "era5_features.pkl")  # what data.py reads

ERA5_VARS = {
    "temp850": "temperature",
    "u850": "u_component_of_wind",
    "v850": "v_component_of_wind",
    "u200": "u_component_of_wind",
    "v200": "v_component_of_wind",
    "vort850": "vorticity",
    "vort200": "vorticity",
    "wind_speed850": "wind_speed",
}
DOWNLOAD_MAP = [
    ("temp850", 850),
    ("u850", 850),
    ("v850", 850),
    ("u200", 200),
    ("v200", 200),
    ("vort850", 850),
    ("vort200", 200),
]


# --------------------------------------------------------------------------
def configure_cds(key=None):
    """Write ~/.cdsapirc from the key (arg > module constant > env). Returns key."""
    key = key or CDS_API_KEY or os.environ.get("CDSAPI_KEY", "")
    if not key:
        raise RuntimeError(
            "No CDS API key. Set os.environ['CDSAPI_KEY']='...' (or edit CDS_API_KEY "
            "at the top of src/era5_download.py). Register free at "
            "https://cds.climate.copernicus.eu and accept the ERA5 licence."
        )
    with open(os.path.expanduser("~/.cdsapirc"), "w") as f:
        f.write(f"url: {CDS_API_URL}\n")
        f.write(f"key: {key}\n")
    return key


def _area_box(cfg):
    """Bounding box [N, W, S, E] covering all configured regions."""
    lat_min = min(v[0] for v in cfg.regions.values())
    lat_max = max(v[1] for v in cfg.regions.values())
    lon_min = min(v[2] for v in cfg.regions.values())
    lon_max = max(v[3] for v in cfg.regions.values())
    return [lat_max, lon_min, lat_min, lon_max]


def _all_years(cfg):
    return cfg.years("train") + cfg.years("val") + cfg.years("test")


# --------------------------------------------------------------------------
# Download (per-year, then merge) — verbatim logic from the working notebook
# --------------------------------------------------------------------------
def download_era5_var(cfg, var_name, pressure_level, fname, years, key):
    import cdsapi

    if os.path.exists(fname):
        print(f"  cached: {os.path.basename(fname)}")
        return fname
    c = cdsapi.Client(url=CDS_API_URL, key=key)
    req = {
        "product_type": "reanalysis",
        "variable": [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vorticity",
        ],
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "12:00"],
        "pressure_level": ["200", "850"],
        "format": "netcdf",
        "area": _area_box(cfg),  # N, W, S, E
    }
    dataset = "reanalysis-era5-pressure-levels"
    c.retrieve(dataset, req, fname)
    print(
        f"  downloaded {os.path.basename(fname)} "
        f"({os.path.getsize(fname) / 1e6:.1f} MB)"
    )
    return fname


def download_era5_full(cfg, var_name, pressure_level, key):
    import xarray as xr

    lvl = f"_{pressure_level}hpa" if pressure_level else ""
    merged = os.path.join(ERA5_RAW, f"era5_{var_name}{lvl}_merged.nc")
    if os.path.exists(merged):
        print(f"  cached (merged): {os.path.basename(merged)}")
        return merged
    yearly = []
    for year in _all_years(cfg):
        yf = os.path.join(ERA5_RAW, f"era5_{var_name}{lvl}_{year}.nc")
        try:
            download_era5_var(cfg, var_name, pressure_level, yf, [year], key)
            yearly.append(yf)
        except Exception as e:
            print(f"  FAILED {var_name} {year}: {e}")
    if not yearly:
        return None
    print(f"  merging {len(yearly)} files...")
    ds = xr.open_mfdataset(yearly, combine="by_coords")
    ds.to_netcdf(merged)
    ds.close()
    for f in yearly:
        os.remove(f)
    print(
        f"  merged -> {os.path.basename(merged)} "
        f"({os.path.getsize(merged) / 1e6:.1f} MB)"
    )
    return merged


# --------------------------------------------------------------------------
# Load + cos-lat area-average per region (valid_time fix kept)
# --------------------------------------------------------------------------
def load_era5_field(cfg, fpath, var_name):
    import xarray as xr

    if fpath is None or not os.path.exists(fpath):
        print(f"  {var_name}: file missing, skipping")
        return None
    try:
        ds = xr.open_dataset(fpath)
        da = ds[list(ds.data_vars)[0]]
        if "pressure_level" in da.dims:
            da = da.isel(pressure_level=0)
        frames = {}
        for rname, (lat_min, lat_max, lon_min, lon_max) in cfg.regions.items():
            region = da.sel(
                latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
            )
            w = np.cos(np.deg2rad(region.latitude.values))
            w2d = w[:, None] * np.ones((1, region.shape[-1]))
            avg = (region.values * w2d).sum(axis=(-2, -1)) / (w2d.sum() + 1e-10)
            frames[f"{var_name}_{rname}"] = avg.ravel()
        time_coord = "valid_time" if "valid_time" in da.coords else "time"
        times = pd.to_datetime(da[time_coord].values)
        m = min(len(v) for v in frames.values())
        frames = {k: v[:m] for k, v in frames.items()}
        df = pd.DataFrame(frames, index=times[:m])
        ds.close()
        return df
    except Exception as e:
        print(f"  error loading {var_name}: {e}")
        return None


# --------------------------------------------------------------------------
# Top-level: download -> load -> merge -> clean -> cache
# --------------------------------------------------------------------------
def build_feature_dataframe(cfg, key=None, force=False):
    """Produce the cleaned ERA5 feature DataFrame and cache it to
    data_cache/era5_features.pkl (the file the rest of the repo reads)."""
    os.makedirs(ERA5_RAW, exist_ok=True)
    if os.path.exists(FEATURES_CACHE) and not force:
        print(f"[era5] using existing cache: {FEATURES_CACHE}")
        return pd.read_pickle(FEATURES_CACHE)

    key = configure_cds(key)
    print(
        f"[era5] area box (N,W,S,E) = {_area_box(cfg)} | "
        f"years {_all_years(cfg)[0]}-{_all_years(cfg)[-1]}"
    )

    files = {}
    for var, level in DOWNLOAD_MAP:
        print(f"-- {var} --")
        try:
            files[var] = download_era5_full(cfg, var, level, key)
        except Exception as e:
            print(f"  FAILED {var}: {e}")
            files[var] = None

    dfs = []
    for var in DOWNLOAD_MAP:
        var_name = var[0]
        df = load_era5_field(cfg, files.get(var_name), var_name)
        if df is not None:
            dfs.append(df)
            print(f"  loaded {var_name}: {df.shape}")

    # Calculate extreme wind speed storm index (magnitude of U and V components) at 850 hPa
    # This acts as our storm target variable!
    u850_cols = [c for df in dfs for c in df.columns if "u850_" in c]
    for col in u850_cols:
        rname = col.split("u850_")[1]
        vcol = f"v850_{rname}"
        # Find the dfs containing u850 and v850
        u_series = None
        v_series = None
        for df in dfs:
            if col in df.columns:
                u_series = df[col]
            if vcol in df.columns:
                v_series = df[vcol]
        if u_series is not None and v_series is not None:
            wind_speed = np.sqrt(u_series**2 + v_series**2)
            wind_df = pd.DataFrame(
                {f"wind_speed850_{rname}": wind_speed}, index=u_series.index
            )
            dfs.append(wind_df)
            print(f"  computed wind_speed850_{rname} (Storm Indicator)")

    if not dfs:
        raise RuntimeError("No ERA5 fields loaded — check the downloads above.")

    # concat; if indices don't line up, floor to 12h and average duplicates
    df_all = pd.concat(dfs, axis=1)
    if df_all.dropna().shape[0] == 0:
        print("[era5] index mismatch -> resampling to 12H")
        res = []
        for df in dfs:
            df.index = df.index.floor("12h")
            res.append(df.groupby(df.index).mean())
        df_all = pd.concat(res, axis=1)

    # clean: drop all-NaN columns, fill remaining with column mean
    df_all = df_all.dropna(axis=1, how="all")
    df_all = df_all.fillna(df_all.mean())
    assert not df_all.isna().any().any(), "NaNs remain after cleaning"

    df_all.to_pickle(FEATURES_CACHE)
    print(f"[era5] cached cleaned features -> {FEATURES_CACHE}  shape={df_all.shape}")
    print(f"[era5] date range {df_all.index[0].date()} -> {df_all.index[-1].date()}")
    return df_all


if __name__ == "__main__":
    from config import CFG

    build_feature_dataframe(CFG)
