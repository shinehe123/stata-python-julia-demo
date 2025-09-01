"""Download and plot OECD %PSE data for Russia vs OECD average.

The script streams data from the OECD's DP_LIVE endpoint, cleans it into a
``pandas`` ``DataFrame`` and saves a comparison plot as a PNG file.  It is
designed to run in non‑interactive environments (CI, terminals without GUI
support, etc.).
"""

from pathlib import Path
import csv
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Use a non-interactive backend for environments without a display
plt.switch_backend("Agg")

URL = (
    "https://stats.oecd.org/sdmx-json/data/DP_LIVE/"
    "AGRSUPP.PSE.PC_GFARM.A?contentType=csv&time=2000:2022"
)


def fetch_oecd_data():
    """Fetch %PSE data for Russia and OECD average from OECD API."""
    with requests.get(URL, timeout=60, stream=True) as resp:
        resp.raise_for_status()
        lines = (line.decode("utf-8") for line in resp.iter_lines())
        reader = csv.DictReader(lines)
        rows = [
            r
            for r in reader
            if r["LOCATION"] in {"RUS", "OECD"} and r.get("FREQUENCY") == "A"
        ]

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[df["OBS_VALUE"].astype(str).str.strip() != ""]
    df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(int)
    df["OBS_VALUE"] = df["OBS_VALUE"].astype(float)
    df = (
        df.pivot_table(
            index="TIME_PERIOD", columns="LOCATION", values="OBS_VALUE", aggfunc="mean"
        ).sort_index()
    )
    return df


def plot_data(df: pd.DataFrame, out_file: Path) -> None:
    """Plot %PSE for Russia and OECD average and save to *out_file*."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["RUS"], marker="o", label="Russia %PSE")
    plt.plot(
        df.index,
        df["OECD"],
        marker="s",
        linestyle="--",
        label="OECD average %PSE",
    )
    plt.title("Russia vs OECD Agricultural Support (2000-2022)")
    plt.xlabel("Year")
    plt.ylabel("%PSE")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Plot saved to {out_file}")


def main():
    df = fetch_oecd_data()
    if df.empty:
        print("未能成功获取数据")
        return

    print(df.describe())
    output = Path(__file__).with_suffix(".png")
    plot_data(df, output)


if __name__ == "__main__":
    main()
