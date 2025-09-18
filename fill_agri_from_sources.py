import argparse
import os
from typing import Dict

import pandas as pd
import requests

# Mapping from crop name to FAOSTAT item codes
CROP_CODES: Dict[str, int] = {
    "wheat": 15,
    "barley": 44,
    "corn": 56,
    "soybeans": 236,
}

# FAOSTAT element codes
PROD_ELEMENT = 5510  # Production quantity
EXPORT_ELEMENT = 5922  # Exports quantity

BASE_URL = "https://faostat-api.apps.fao.org/api/v1/en/"


def fetch_faostat(dataset: str, params: Dict[str, int]):
    """Fetch a single value from FAOSTAT."""
    url = BASE_URL + dataset
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return None
    return float(data[0]["value"])


def fetch_production(area: int, item: int, year: int):
    params = {
        "area_code": area,
        "item_code": item,
        "element_code": PROD_ELEMENT,
        "year": year,
    }
    return fetch_faostat("QCL/Production_Crops_Livestock_E_A", params)


def fetch_exports(area: int, year: int):
    params = {
        "area_code": area,
        "item_code": CROP_CODES["wheat"],
        "element_code": EXPORT_ELEMENT,
        "year": year,
    }
    return fetch_faostat("TP/TP_TRADE_E_A", params)


def main(infile: str, outbase: str):
    df = pd.read_csv(infile)
    if "area_code" not in df.columns or "year" not in df.columns:
        raise ValueError("Input file must contain 'area_code' and 'year' columns")

    # Fetch production for each crop
    for crop, code in CROP_CODES.items():
        df[f"{crop}_prod"] = [
            fetch_production(area, code, year)
            for area, year in zip(df["area_code"], df["year"])
        ]

    # Wheat exports
    exports = None
    comtrade_path = "comtrade_wheat_exports.csv"
    if os.path.exists(comtrade_path):
        exports = pd.read_csv(comtrade_path)
        exports = exports[["area_code", "year", "wheat_exports_comtrade"]]
        df = df.merge(exports, on=["area_code", "year"], how="left")

    df["wheat_exports"] = df.get("wheat_exports_comtrade")
    mask = df["wheat_exports"].isna()
    if mask.any():
        df.loc[mask, "wheat_exports"] = [
            fetch_exports(area, year)
            for area, year in zip(df.loc[mask, "area_code"], df.loc[mask, "year"])
        ]
    if "wheat_exports_comtrade" in df.columns:
        df = df.drop(columns=["wheat_exports_comtrade"])

    # Optional PSD merge
    psd_path = "psd_production_subset.csv"
    if os.path.exists(psd_path):
        psd = pd.read_csv(psd_path)
        df = df.merge(psd, on=["area_code", "year"], how="left", suffixes=("", "_psd"))

    df.to_csv(f"{outbase}.csv", index=False)
    df.to_excel(f"{outbase}.xlsx", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill production and trade data from multiple sources.")
    parser.add_argument("--input", required=True, help="Input CSV with 'area_code' and 'year'.")
    parser.add_argument("--outbase", default="agri_filled", help="Base name for output files.")
    args = parser.parse_args()
    main(args.input, args.outbase)
