import requests
import pandas as pd

def fetch_power_annual(lat: float, lon: float, start: int, end: int) -> pd.DataFrame:
    """Fetch annual average temperature and precipitation from NASA POWER."""
    params = {
        "parameters": "T2M,PRECTOT",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    url = "https://power.larc.nasa.gov/api/temporal/annual/point"
    try:
        r = requests.get(url, params=params, timeout=1)
        r.raise_for_status()
        data = r.json()["properties"]["parameter"]
        years = sorted(int(y) for y in data["T2M"].keys())
        df = pd.DataFrame({
            "year": years,
            "tavg_c": [data["T2M"][str(y)] for y in years],
            "precip_mm": [data["PRECTOT"][str(y)] for y in years],
        })
        return df
    except Exception:
        return pd.DataFrame(columns=["year", "tavg_c", "precip_mm"])

def compute_gdd(df: pd.DataFrame, base: float = 10.0) -> pd.DataFrame:
    out = df.copy()
    out["gdd_base10"] = (out["tavg_c"] - base).clip(lower=0) * 365
    return out
