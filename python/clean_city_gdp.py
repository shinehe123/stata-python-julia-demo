import pandas as pd
import argparse


def clean_and_compute(input_path: str, output_path: str = None) -> pd.DataFrame:
    """Load city GDP data, drop missing values, and compute per-capita GDP.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file. The file should contain the columns
        ``city``, ``gdp`` and ``population``.
    output_path : str, optional
        If provided, the cleaned data with the new ``gdp_per_capita`` column
        will be written to this CSV file.

    Returns
    -------
    pandas.DataFrame
        The cleaned DataFrame with an additional ``gdp_per_capita`` column.
    """
    df = pd.read_csv(input_path)
    # Ensure expected columns exist
    expected_cols = {"city", "gdp", "population"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Drop rows with any missing values in the key columns
    df = df.dropna(subset=["city", "gdp", "population"])

    # Compute per capita GDP; avoid division by zero
    df = df.assign(gdp_per_capita=df["gdp"] / df["population"].replace({0: pd.NA}))

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="Clean city GDP data and compute per-capita GDP")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output CSV file path", default=None)
    args = parser.parse_args()

    df = clean_and_compute(args.input, args.output)
    print(df)


if __name__ == "__main__":
    main()
