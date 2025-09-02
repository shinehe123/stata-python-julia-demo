import requests
import pandas as pd
from typing import Dict

# Common item codes for crops
ITEMS: Dict[str, int] = {
    "wheat": 15,
    "barley": 44,
    "maize": 56,
    "soybeans": 236,
}

# Element codes for production, area, yield in the QCL domain
ELEMENTS_QCL: Dict[str, int] = {
    "production": 5510,
    "area_harvested": 5312,
    "yield": 5419,
}

# FAOSTAT area code for Russia
AREA_RUS = 185

BASE_URL = "https://fenixservices.fao.org/faostat/api/v1/en"

def _faostat_request(domain: str, params: Dict[str, str]) -> pd.DataFrame:
    url = f"{BASE_URL}/{domain}/FAOSTAT_DATA"
    try:
        r = requests.get(url, params=params, timeout=1)
        r.raise_for_status()
        data = r.json().get("data", [])
        return pd.DataFrame(data)
    except Exception:
        # Return empty on network errors so downstream code can handle missing data
        return pd.DataFrame()

def fetch_qcl(item_code: int, element_code: int, area_code: int, start_year: int, end_year: int) -> pd.DataFrame:
    params = {
        "item_code": item_code,
        "element_code": element_code,
        "area_code": area_code,
        "year": f"{start_year}:{end_year}",
    }
    df = _faostat_request("QCL", params)
    if df.empty:
        return pd.DataFrame(columns=["year", "value"])
    return df[["year", "value"]].assign(year=lambda d: d["year"].astype(int), value=lambda d: pd.to_numeric(d["value"], errors="coerce"))

def fetch_trade_tm(item_code: int, element_code: int, area_code: int, start_year: int, end_year: int) -> pd.DataFrame:
    params = {
        "item_code": item_code,
        "element_code": element_code,
        "reporting_area_code": area_code,
        "year": f"{start_year}:{end_year}",
    }
    df = _faostat_request("TM", params)
    if df.empty:
        return pd.DataFrame(columns=["year", "value"])
    return df[["year", "value"]].assign(year=lambda d: d["year"].astype(int), value=lambda d: pd.to_numeric(d["value"], errors="coerce"))
