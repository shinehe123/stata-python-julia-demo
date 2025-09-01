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


def plot_data(df: pd.DataFrame) -> None:
    """Plot %PSE for Russia and OECD average."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["RUS"], marker="o", label="俄罗斯 %PSE")
    plt.plot(df.index, df["OECD"], marker="s", linestyle="--", label="OECD国家平均 %PSE")
    plt.title("俄罗斯与OECD国家平均农业支持率对比 (2000-2022)")
    plt.xlabel("年份")
    plt.ylabel("%PSE (占农业总产值百分比)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    df = fetch_oecd_data()
    if df.empty:
        print("未能成功获取数据")
        return

    print(df.describe())
    plot_data(df)


if __name__ == "__main__":
    main()
