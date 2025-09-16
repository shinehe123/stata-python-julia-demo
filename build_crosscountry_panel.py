# -*- coding: utf-8 -*-
"""Build cross-country panel for SCM/DID using FAOSTAT + World Bank."""
import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

from src.sources.faostat import (
    fetch_qcl,
    fetch_tm_quantity,
    ITEMS,
    ELEMENTS_QCL,
    ELEMENTS_TM,
)
from src.sources.worldbank import fetch_wb_multi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

START_YEAR = 2000
END_YEAR = 2024

DONOR_COUNTRIES: List[Tuple[str, str, int]] = [
    ("AUS", "Australia", 8), ("CAN", "Canada", 124), ("ARG", "Argentina", 10),
    ("KAZ", "Kazakhstan", 398), ("UKR", "Ukraine", 804), ("ZAF", "South Africa", 710),
    ("USA", "United States", 840), ("FRA", "France", 250)
]

RUS_COUNTRY: Tuple[str, str, int] = ("RUS", "Russia", 643)

WB_INDICATORS: Dict[str, str] = {
    "PA.NUS.FCRF": "exchange_rate_local_per_usd",
    "NY.GDP.PCAP.KD": "gdp_per_capita_usd_const2015",
    "FP.CPI.TOTL": "cpi_2015_100",
}


def _fetch_country(iso3c: str, name: str, fao_code: int) -> pd.DataFrame:
    years = pd.DataFrame({"year": range(START_YEAR, END_YEAR + 1)})

    # Production for key crops
    w_prod = fetch_qcl(ITEMS["wheat"], ELEMENTS_QCL["production"], fao_code, START_YEAR, END_YEAR)
    b_prod = fetch_qcl(ITEMS["barley"], ELEMENTS_QCL["production"], fao_code, START_YEAR, END_YEAR)
    m_prod = fetch_qcl(ITEMS["maize"], ELEMENTS_QCL["production"], fao_code, START_YEAR, END_YEAR)
    s_prod = fetch_qcl(ITEMS["soybeans"], ELEMENTS_QCL["production"], fao_code, START_YEAR, END_YEAR)

    df = years.copy()
    df["wheat_production_tonnes"] = w_prod.set_index("year")["value"].reindex(df["year"]).values
    df["barley_production_tonnes"] = b_prod.set_index("year")["value"].reindex(df["year"]).values
    df["corn_production_tonnes"] = m_prod.set_index("year")["value"].reindex(df["year"]).values
    df["soybean_production_tonnes"] = s_prod.set_index("year")["value"].reindex(df["year"]).values

    # Trade (wheat exports)
    w_exp = fetch_tm_quantity(
        ITEMS["wheat"],
        ELEMENTS_TM["export_qty"],
        fao_code,
        START_YEAR,
        END_YEAR,
    )
    df["wheat_export_tonnes"] = w_exp.set_index("year")["value"].reindex(df["year"]).values

    # World Bank indicators
    wb = fetch_wb_multi(iso3c, list(WB_INDICATORS.keys()), START_YEAR, END_YEAR)
    if not wb.empty:
        wb = wb.rename(columns=WB_INDICATORS)
        df = df.merge(wb, on="year", how="left")
    else:
        for v in WB_INDICATORS.values():
            df[v] = None

    df.insert(0, "iso3c", iso3c)
    df.insert(1, "country_name", name)
    return df


def main(out_csv: Path, schema_csv: Path) -> None:
    schema_cols = list(pd.read_csv(schema_csv, nrows=0).columns)
    countries = [RUS_COUNTRY] + DONOR_COUNTRIES
    frames = [_fetch_country(*c) for c in countries]
    panel = pd.concat(frames, ignore_index=True)

    # Ensure all schema columns present
    for col in schema_cols:
        if col not in panel.columns:
            panel[col] = None

    panel = panel[schema_cols].sort_values(["iso3c", "year"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_csv, index=False)
    logging.info("Saved %s", out_csv)


if __name__ == "__main__":
    output_path = Path("output/crosscountry_panel_for_scm.csv")
    schema_path = Path("data/schemas/schema_crosscountry_scm_panel.csv")
    output_path.parent.mkdir(exist_ok=True)
    main(output_path, schema_path)
