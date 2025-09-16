import pandas as pd
from pathlib import Path

def load_psd_local(path: Path, country: str, commodity: str) -> pd.DataFrame:
    """Load USDA PSD CSV and filter for country and commodity."""
    df = pd.read_csv(path)
    df = df[(df["Country"] == country) & (df["Commodity"] == commodity)]
    if "Market Year" in df.columns:
        df = df.rename(columns={"Market Year": "year"})
    return df[["year", "Production", "Imports", "Exports", "Domestic Consumption", "Ending Stocks"]]
