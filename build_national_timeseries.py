# -*- coding: utf-8 -*-
"""Build Russia national time series according to schema."""
import pandas as pd
from pathlib import Path
from src.sources.worldbank import fetch_wb_multi
from src.sources.faostat import (
    fetch_qcl,
    fetch_tm_quantity,
    ITEMS,
    ELEMENTS_QCL,
    ELEMENTS_TM,
    AREA_RUS,
)
from src.sources.nasa_power import fetch_power_annual, compute_gdd
from src.sources.usda_psd import load_psd_local

START, END = 2000, 2024

def main(out_csv: Path, schema_csv: Path, psd_csv: Path = None):
    # Start from empty schema
    schema_cols = list(pd.read_csv(schema_csv, nrows=0).columns)
    df = pd.DataFrame(columns=schema_cols)

    # --- FAOSTAT Production/Area/Yield ---
    # wheat
    w_prod = fetch_qcl(ITEMS["wheat"], ELEMENTS_QCL["production"], AREA_RUS, START, END)
    w_area = fetch_qcl(ITEMS["wheat"], ELEMENTS_QCL["area_harvested"], AREA_RUS, START, END)
    w_yield = fetch_qcl(ITEMS["wheat"], ELEMENTS_QCL["yield"], AREA_RUS, START, END)

    # barley, maize, soy
    b_prod = fetch_qcl(ITEMS["barley"], ELEMENTS_QCL["production"], AREA_RUS, START, END)
    m_prod = fetch_qcl(ITEMS["maize"], ELEMENTS_QCL["production"], AREA_RUS, START, END)
    s_prod = fetch_qcl(ITEMS["soybeans"], ELEMENTS_QCL["production"], AREA_RUS, START, END)

    # Map into our columns
    prod = pd.DataFrame({"year": w_prod["year"]}).drop_duplicates()
    prod["wheat_production_tonnes"] = w_prod["value"].values
    prod["barley_production_tonnes"] = b_prod.set_index("year")["value"].reindex(prod["year"]).values
    prod["corn_production_tonnes"] = m_prod.set_index("year")["value"].reindex(prod["year"]).values
    prod["soybean_production_tonnes"] = s_prod.set_index("year")["value"].reindex(prod["year"]).values
    # Harvested area & yield (use wheat as proxy for harvested area/yield if cereals total unavailable)
    prod["harvested_area_thousand_ha"] = w_area.set_index("year")["value"].reindex(prod["year"]).values / 1000.0
    prod["grain_yield_t_per_ha"] = w_yield.set_index("year")["value"].reindex(prod["year"]).values

    # --- FAOSTAT Trade (TM) for wheat exports/imports ---
    w_exp = fetch_tm_quantity(ITEMS["wheat"], ELEMENTS_TM["export_qty"], AREA_RUS, START, END)
    w_imp = fetch_tm_quantity(ITEMS["wheat"], ELEMENTS_TM["import_qty"], AREA_RUS, START, END)
    prod["wheat_export_tonnes"] = w_exp.set_index("year")["value"].reindex(prod["year"]).values
    prod["grain_export_tonnes"] = prod["wheat_export_tonnes"]
    prod["grain_import_tonnes"] = w_imp.set_index("year")["value"].reindex(prod["year"]).values

    # --- World Bank indicators ---
    wb = fetch_wb_multi(
        "RUS",
        indicators=[
            "PA.NUS.FCRF",
            "EP.PMP.DESL.CD",
            "SP.RUR.TOTL",
        ],
        start=START,
        end=END,
    ).rename(
        columns={
            "PA.NUS.FCRF": "exchange_rate_rub_per_usd",
            "EP.PMP.DESL.CD": "diesel_price_usd_per_l",
            "SP.RUR.TOTL": "rural_population_million",
        }
    )
    if not wb.empty:
        wb["rural_population_million"] = wb["rural_population_million"] / 1e6
    for col in ["exchange_rate_rub_per_usd", "diesel_price_usd_per_l", "rural_population_million"]:
        if col not in wb.columns:
            wb[col] = None

    # --- NASA POWER climate ---
    clim = fetch_power_annual(55.0, 37.0, START, END)
    clim = compute_gdd(clim).rename(
        columns={
            "precip_mm": "growing_season_precip_mm",
            "gdd_base10": "growing_degree_days_base_10c",
        }
    )[["year", "growing_season_precip_mm", "growing_degree_days_base_10c"]]

    # --- Merge ---
    out = (
        prod.merge(
            wb[
                [
                    "year",
                    "exchange_rate_rub_per_usd",
                    "diesel_price_usd_per_l",
                    "rural_population_million",
                ]
            ],
            on="year",
            how="left",
        ).merge(clim, on="year", how="left")
    )

    # --- Derived fields & dummies ---
    out["covid_dummy"] = (out["year"] >= 2020).astype(int)
    for k in [2010, 2012, 2014, 2020]:
        out[f"policy{k}_dummy"] = (out["year"] == k).astype(int)
    out["post2014_dummy"] = (out["year"] >= 2014).astype(int)
    out["time_trend"] = out["year"] - out["year"].min()
    out["time_after2014_trend"] = (out["year"] - 2014).clip(lower=0)

    # --- Optional USDA PSD ---
    if psd_csv and Path(psd_csv).exists():
        psd = load_psd_local(Path(psd_csv), country="Russia", commodity="Wheat")
        psd = psd.rename(
            columns={
                "Production": "psd_wheat_production_tonnes",
                "Imports": "psd_wheat_imports_tonnes",
                "Exports": "psd_wheat_exports_tonnes",
                "Domestic Consumption": "psd_domestic_consumption_tonnes",
                "Ending Stocks": "psd_ending_stocks_tonnes",
            }
        )
        out = out.merge(psd, left_on="year", right_on="year", how="left")
        out["stocks_to_use_ratio_pct"] = (
            100.0 * out["psd_ending_stocks_tonnes"] / out["psd_domestic_consumption_tonnes"]
        )
        out["self_sufficiency_ratio_pct"] = (
            100.0 * out["psd_wheat_production_tonnes"] / out["psd_domestic_consumption_tonnes"]
        )
        out["import_dependency_ratio_pct"] = (
            100.0 * out["psd_wheat_imports_tonnes"] / out["psd_domestic_consumption_tonnes"]
        )
    else:
        out["stocks_to_use_ratio_pct"] = None
        out["self_sufficiency_ratio_pct"] = None
        out["import_dependency_ratio_pct"] = None

    # --- Conform to schema columns & save ---
    for col in [c for c in schema_cols if c not in out.columns]:
        out[col] = None
    out = out[schema_cols].sort_values("year").reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    out_csv = Path("output/russia_national_timeseries.csv")
    schema_csv = Path("data/schemas/schema_national_rus_timeseries.csv")
    psd_csv = Path("data/external/usda_psd.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    main(out_csv, schema_csv, psd_csv)
