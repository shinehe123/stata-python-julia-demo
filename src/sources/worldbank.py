import requests
import pandas as pd
from typing import List, Dict

def fetch_wb_multi(country_code: str, indicators: List[str], start: int, end: int) -> pd.DataFrame:
    """Fetch multiple World Bank indicators for a given country."""
    frames: List[pd.DataFrame] = []
    for ind in indicators:
        url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{ind}"
        params = {"format": "json", "per_page": 20000, "date": f"{start}:{end}"}
        try:
            r = requests.get(url, params=params, timeout=1)
            r.raise_for_status()
            json_data = r.json()
            if len(json_data) < 2 or json_data[1] is None:
                continue
            data = json_data[1]
            df = pd.DataFrame(
                {"year": [int(d["date"]) for d in data], ind: [d["value"] for d in data]}
            )
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="year", how="outer")
    return out.sort_values("year").reset_index(drop=True)
