import requests, time
import pandas as pd
from typing import Dict, Optional
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

BASE_URL = "https://fenixservices.fao.org/faostat/api/v1/en"

# Crop item codes
ITEMS: Dict[str, int] = {"wheat": 15, "barley": 44, "maize": 56, "soybeans": 236}
# QCL element codes
ELEMENTS_QCL: Dict[str, int] = {"production": 5510, "area_harvested": 5312, "yield": 5419}
# Trade matrix element codes
ELEMENTS_TM: Dict[str, int] = {
    "import_qty": 5610,
    "export_qty": 5910,
    # amounts: "import_val": 5622, "export_val": 5922
}

AREA_RUS = 185  # Russia

def _mk_session(total_retries: int = 4, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "research-script/1.0 (contact: you@example.com)"})
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

SESSION = _mk_session()

def _faostat_request(domain: str, params: Dict[str, str], timeout: int = 30) -> pd.DataFrame:
    url = f"{BASE_URL}/{domain}/FAOSTAT_DATA"
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json().get("data", [])
        return pd.DataFrame(data)
    except Exception:
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
        return pd.DataFrame(columns=["year", "value", "unit"])
    df = df[["year", "value", "unit"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["year"])

def fetch_tm_quantity(item_code: int, element_code: int, reporting_area_code: int,
                      start_year: int, end_year: int,
                      partner_area_code: Optional[int] = None) -> pd.DataFrame:
    params = {
        "item_code": item_code,
        "element_code": element_code,
        "reporting_area_code": reporting_area_code,
        "year": f"{start_year}:{end_year}",
    }
    if partner_area_code is not None:
        params["partner_area_code"] = partner_area_code
    df = _faostat_request("TM", params)
    if df.empty:
        return pd.DataFrame(columns=["year", "value"])
    df = df[["year", "value"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return (df.dropna(subset=["year"]).groupby("year", as_index=False)["value"].sum())

def build_russia_grains(start_year: int = 2010, end_year: int = 2023) -> pd.DataFrame:
    pieces = []
    for crop, code in ITEMS.items():
        prod = fetch_qcl(code, ELEMENTS_QCL["production"], AREA_RUS, start_year, end_year).rename(columns={"value": f"{crop}_production_t"})
        area = fetch_qcl(code, ELEMENTS_QCL["area_harvested"], AREA_RUS, start_year, end_year).rename(columns={"value": f"{crop}_area_ha"})
        yld = fetch_qcl(code, ELEMENTS_QCL["yield"], AREA_RUS, start_year, end_year).rename(columns={"value": f"{crop}_yield_hg_per_ha"})
        df = prod.merge(area[["year", f"{crop}_area_ha"]], on="year", how="outer").merge(yld[["year", f"{crop}_yield_hg_per_ha"]], on="year", how="outer")
        pieces.append(df)
    qcl_wide = pieces[0]
    for df in pieces[1:]:
        qcl_wide = qcl_wide.merge(df, on="year", how="outer")
    for crop in ITEMS:
        if f"{crop}_yield_hg_per_ha" in qcl_wide:
            qcl_wide[f"{crop}_yield_t_per_ha"] = qcl_wide[f"{crop}_yield_hg_per_ha"] / 10000.0
            qcl_wide.drop(columns=[f"{crop}_yield_hg_per_ha"], inplace=True)
    tm_exports, tm_imports = [], []
    for crop, code in ITEMS.items():
        ex = fetch_tm_quantity(code, ELEMENTS_TM["export_qty"], AREA_RUS, start_year, end_year).rename(columns={"value": f"{crop}_export_t"})
        im = fetch_tm_quantity(code, ELEMENTS_TM["import_qty"], AREA_RUS, start_year, end_year).rename(columns={"value": f"{crop}_import_t"})
        tm_exports.append(ex); tm_imports.append(im)
    tm_exp = tm_exports[0]
    for df in tm_exports[1:]:
        tm_exp = tm_exp.merge(df, on="year", how="outer")
    tm_imp = tm_imports[0]
    for df in tm_imports[1:]:
        tm_imp = tm_imp.merge(df, on="year", how="outer")
    for df in (tm_exp, tm_imp):
        for col in df.columns:
            if col != "year":
                df[col] = pd.to_numeric(df[col], errors="coerce")
    tm_exp["grain_export_tonnes"] = tm_exp.drop(columns=["year"]).sum(axis=1, skipna=True)
    tm_imp["grain_import_tonnes"] = tm_imp.drop(columns=["year"]).sum(axis=1, skipna=True)
    out = qcl_wide.merge(tm_exp[["year", "grain_export_tonnes"]], on="year", how="left").merge(tm_imp[["year", "grain_import_tonnes"]], on="year", how="left")
    out = out.rename(columns={
        "maize_production_t": "corn_production_tonnes",
        "maize_area_ha": "corn_area_ha",
        "maize_yield_t_per_ha": "corn_yield_t_per_ha",
        "wheat_production_t": "wheat_production_tonnes",
        "barley_production_t": "barley_production_tonnes",
        "soybeans_production_t": "soybean_production_tonnes",
    })
    area_cols = [c for c in out.columns if c.endswith("_area_ha")]
    out["harvested_area_thousand_ha"] = out[area_cols].sum(axis=1, skipna=True) / 1000.0
    for crop in ("wheat", "barley", "corn", "soybean"):
        ycol = f"{crop}_yield_t_per_ha"
        pcol = f"{crop}_production_tonnes" if crop != "corn" else "corn_production_tonnes"
        acol = f"{crop}_area_ha"
        if ycol not in out and pcol in out and acol in out:
            out[ycol] = out[pcol] / out[acol].replace(0, pd.NA)
    num = sum(out.get(f"{c}_yield_t_per_ha", 0) * out.get(f"{c}_area_ha", 0) for c in ("wheat", "barley", "corn", "soybean"))
    den = sum(out.get(f"{c}_area_ha", 0) for c in ("wheat", "barley", "corn", "soybean"))
    out["grain_yield_t_per_ha"] = num / den.replace(0, pd.NA)
    keep = ["year",
            "wheat_production_tonnes","barley_production_tonnes","corn_production_tonnes","soybean_production_tonnes",
            "harvested_area_thousand_ha","grain_yield_t_per_ha",
            "grain_export_tonnes","grain_import_tonnes"]
    return out.sort_values("year").reset_index(drop=True)[keep]
