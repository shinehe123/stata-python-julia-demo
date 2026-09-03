"""Command-line interface."""

from __future__ import annotations

import argparse
from itertools import chain
from pathlib import Path

from .pipeline import ALIASES, build_tables, iter_records, load_companies, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 CNIPA PSS 著录项导出文件整理为上市公司专利合作数据"
    )
    parser.add_argument("--companies", required=True, type=Path, help="上市公司及别名 CSV")
    parser.add_argument("--input", required=True, nargs="+", type=Path, help="PSS 导出文件")
    parser.add_argument("--output-dir", default=Path("output"), type=Path)
    args = parser.parse_args()

    records = chain.from_iterable(iter_records(path) for path in args.input)
    patents, edges = build_tables(records, load_companies(args.companies))
    patent_fields = [
        "patent_id", "company_id", "company_name", *ALIASES,
        "applicant_count", "is_collaboration",
    ]
    write_csv(args.output_dir / "company_patents.csv", patents, patent_fields)
    write_csv(
        args.output_dir / "collaboration_edges.csv",
        edges,
        ["company_id", "partner_name", "year", "joint_patent_count"],
    )
    print(f"写入 {len(patents)} 条公司-专利记录和 {len(edges)} 条合作边")


if __name__ == "__main__":
    main()

