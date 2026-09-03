"""Export co-applicant patents from an authorised Patsnap-compatible API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_ENDPOINT = "https://connect.zhihuiya.com/search/patent/query-search-patent/v2"


@dataclass(frozen=True)
class Company:
    code: str
    name: str
    aliases: tuple[str, ...]
    address: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.name, *self.aliases)))


def read_companies(path: Path) -> list[Company]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"code", "name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("公司文件必须包含 code 和 name 列")
        companies = []
        for row in reader:
            aliases = tuple(x.strip() for x in row.get("aliases", "").split("|") if x.strip())
            if row["code"].strip() and row["name"].strip():
                companies.append(Company(row["code"].strip(), row["name"].strip(), aliases,
                                         row.get("address", "").strip()))
    return companies


def nested(value: Any, dotted_path: str) -> Any:
    for part in dotted_path.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        else:
            value = value[part]
    return value


def optional_nested(value: Any, dotted_path: str) -> Any:
    """Read an optional response field without hiding malformed configured paths."""
    try:
        return nested(value, dotted_path)
    except (KeyError, IndexError):
        return ""


class APIClient:
    def __init__(self, endpoint: str, api_key: str, timeout: float, retries: int) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode()
        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "stata-python-julia-demo/1.0",
            },
        )
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                    if result.get("status") is False:
                        raise RuntimeError(
                            f"API 业务错误 {result.get('error_code')}: {result.get('error_msg', '未知错误')}"
                        )
                    return result
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.retries:
                    detail = exc.read().decode("utf-8", "replace")[:500]
                    raise RuntimeError(f"API 返回 HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                if attempt == self.retries:
                    raise RuntimeError(f"无法连接 API: {exc.reason}") from exc
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")


def normalize_assignees(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return list(dict.fromkeys(part.strip() for part in raw.split("|") if part.strip()))
    result = []
    for item in raw:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("name") or item.get("original_name") or item.get("value") or ""
        else:
            name = ""
        if str(name).strip():
            result.append(str(name).strip())
    return list(dict.fromkeys(result))


def normalize_name(value: str) -> str:
    """Normalize punctuation and whitespace for exact organization matching."""
    return "".join(str(value).strip().replace("（", "(").replace("）", ")").split()).casefold()


def normalize_values(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split("|") if part.strip()]
    if isinstance(raw, list):
        return [str(item.get("value", item.get("address", "")) if isinstance(item, dict) else item).strip()
                for item in raw if str(item).strip()]
    return [str(raw).strip()]


def read_repaco(path: Path) -> dict[str, set[str]]:
    """Read a Repaco-derived code/name membership table."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"code", "name"}.issubset(reader.fieldnames):
            raise ValueError("Repaco 文件必须包含 code 和 name 列")
        groups: dict[str, set[str]] = {}
        for row in reader:
            code, name = row["code"].strip(), row["name"].strip()
            if code and name:
                groups.setdefault(code, set()).add(normalize_name(name))
        return groups


def infer_patent_type(publication_number: str) -> str:
    match = re.search(r"([ABUS])\d?$", publication_number.strip().upper())
    kind = match.group(1) if match else ""
    return {"A": "发明申请", "B": "发明授权", "U": "实用新型", "S": "外观设计"}.get(kind, "")


def query_for(company: Company, template: str) -> str:
    expression = " OR ".join(f'\"{name}\"' for name in company.names)
    return template.format(company=expression)


def patent_rows(
    company: Company,
    records: Iterable[dict[str, Any]],
    fields: dict[str, str],
    group_names: set[str] | None = None,
) -> Iterable[dict[str, str]]:
    own_names = {normalize_name(name) for name in company.names}
    excluded_names = own_names | (group_names or set())
    for record in records:
        assignees = normalize_assignees(nested(record, fields["assignees"]))
        if len(assignees) < 2:
            continue
        normalized_assignees = [normalize_name(name) for name in assignees]
        if own_names.isdisjoint(normalized_assignees):
            continue
        partners = [name for name, normalized in zip(assignees, normalized_assignees)
                    if normalized not in excluded_names]
        if not partners:
            continue
        publication_number = str(nested(record, fields["publication_number"]))
        addresses = normalize_values(optional_nested(record, fields["applicant_addresses"]))
        # P002 summary records omit addresses. Preserve positional alignment and
        # fill the matched applicant with the listed-company address.
        addresses = (addresses + [""] * len(assignees))[:len(assignees)]
        for index, name in enumerate(normalized_assignees):
            if not addresses[index] and name in own_names:
                addresses[index] = company.address
        listed_addresses = [address for name, address in zip(normalized_assignees, addresses)
                            if name in own_names and address]
        patent_type = str(optional_nested(record, fields["patent_type"]))
        common = {
            "listed_company_code": company.code,
            "listed_company_name": company.name,
            "publication_number": publication_number,
            "application_number": str(optional_nested(record, fields["application_number"])),
            "title": str(nested(record, fields["title"])),
            "application_date": str(nested(record, fields["application_date"])),
            "patent_type": patent_type or infer_patent_type(publication_number),
            "all_assignees": "|".join(assignees),
            "applicant_addresses": "|".join(addresses),
            "listed_company_addresses": "|".join(dict.fromkeys(listed_addresses)),
        }
        for partner in partners:
            yield {**common, "partner_name": partner}


def parse_field_mapping(values: list[str]) -> dict[str, str]:
    mapping = {
        "publication_number": "pn",
        "application_number": "apno",
        "title": "title",
        "application_date": "apdt",
        "assignees": "original_assignee",
        "patent_type": "patent_type",
        "applicant_addresses": "address",
    }
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key not in mapping or not path:
            raise ValueError(f"无效字段映射: {value}")
        mapping[key] = path
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过获授权的 API 导出上市公司共同申请专利")
    parser.add_argument("--companies", type=Path, required=True, help="上市公司 CSV")
    parser.add_argument("--output", type=Path, required=True, help="输出明细 CSV")
    parser.add_argument("--endpoint", default=os.getenv("PATENT_API_ENDPOINT", DEFAULT_ENDPOINT), help="专利检索 API URL")
    parser.add_argument("--api-key-env", default="PATENT_API_KEY", help="存放密钥的环境变量名")
    parser.add_argument("--query-template", default="ANS:({company})")
    parser.add_argument("--query-key", default="query_text")
    parser.add_argument("--page-key", default="offset")
    parser.add_argument("--page-size-key", default="limit")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--results-path", default="data.results")
    parser.add_argument("--total-path", default="data.total_search_result_count")
    parser.add_argument("--field", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--repaco", type=Path, help="Repaco 同集团字典 CSV（code,name）")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2, help="每次请求后的等待秒数")
    return parser


def run(args: argparse.Namespace) -> int:
    api_key = os.getenv(args.api_key_env)
    if not args.endpoint:
        raise ValueError("请通过 --endpoint 或 PATENT_API_ENDPOINT 提供 API 地址")
    if not api_key:
        raise ValueError(f"环境变量 {args.api_key_env} 未设置")
    if args.page_size < 1 or args.retries < 0 or args.sleep < 0:
        raise ValueError("分页、重试或等待参数无效")

    companies = read_companies(args.companies)
    groups = read_repaco(args.repaco) if args.repaco else {}
    # Rows sharing a parent stock code (for example subsidiaries) are also same-group names.
    for company in companies:
        groups.setdefault(company.code, set()).update(normalize_name(name) for name in company.names)
    fields = parse_field_mapping(args.field)
    client = APIClient(args.endpoint, api_key, args.timeout, args.retries)
    columns = [
        "listed_company_code", "listed_company_name", "partner_name",
        "publication_number", "application_number", "title", "application_date",
        "patent_type", "all_assignees", "applicant_addresses", "listed_company_addresses",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, str]] = set()
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, company in enumerate(companies, 1):
            page = 1
            exported = 0
            while True:
                payload = {
                    args.query_key: query_for(company, args.query_template),
                    args.page_key: (page - 1) * args.page_size,
                    args.page_size_key: args.page_size,
                }
                response = client.post(payload)
                records = nested(response, args.results_path)
                total = int(nested(response, args.total_path))
                if not isinstance(records, list):
                    raise ValueError(f"{args.results_path} 不是数组")
                for row in patent_rows(company, records, fields, groups.get(company.code)):
                    identity = (row["listed_company_code"], row["partner_name"], row["publication_number"])
                    if identity not in seen:
                        writer.writerow(row)
                        seen.add(identity)
                        exported += 1
                print(f"[{index}/{len(companies)}] {company.code} 第 {page} 页，累计 {exported} 条", file=sys.stderr)
                if not records or page * args.page_size >= total:
                    break
                page += 1
                time.sleep(args.sleep)
    return len(seen)


def main() -> None:
    try:
        count = run(build_parser().parse_args())
    except (ValueError, KeyError, IndexError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"完成：导出 {count} 条公司—合作方—专利记录")


if __name__ == "__main__":
    main()
