import csv
import tempfile
import unittest
from pathlib import Path

from patent_collaboration.cli import (
    Company,
    infer_patent_type,
    normalize_assignees,
    parse_field_mapping,
    patent_rows,
    query_for,
    read_companies,
    read_repaco,
)


class CollaborationTests(unittest.TestCase):
    def test_reads_utf8_bom_and_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "companies.csv"
            path.write_text("\ufeffcode,name,aliases\n1,甲公司,甲|甲股份\n", encoding="utf-8")
            self.assertEqual(read_companies(path), [Company("1", "甲公司", ("甲", "甲股份"))])

    def test_reads_and_fills_listed_company_address(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "companies.csv"
            path.write_text("code,name,aliases,address\n1,甲公司,,北京市一号\n", encoding="utf-8")
            company = read_companies(path)[0]
        record = {"pn": "CN1A", "title": "方法", "apdt": 20200101,
                  "original_assignee": "甲公司|乙大学"}
        rows = list(patent_rows(company, [record], parse_field_mapping([])))
        self.assertEqual(rows[0]["applicant_addresses"], "北京市一号|")
        self.assertEqual(rows[0]["listed_company_addresses"], "北京市一号")

    def test_query_quotes_each_company_name(self):
        company = Company("1", "甲公司", ("甲股份",))
        self.assertEqual(query_for(company, "AP=({company})"), 'AP=("甲公司" OR "甲股份")')

    def test_emits_only_collaboration_partners_and_deduplicates_assignees(self):
        records = [{
            "publication_number": "CN123A",
            "title": "共同发明",
            "application_date": "2020-01-02",
            "assignees": [{"name": "甲公司"}, {"name": "乙大学"}, {"name": "乙大学"}],
        }]
        fields = {key: key for key in ("publication_number", "title", "application_date", "assignees")}
        fields.update(application_number="application_number", patent_type="patent_type",
                      applicant_addresses="applicant_addresses")
        rows = list(patent_rows(Company("1", "甲公司", ()), records, fields))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["partner_name"], "乙大学")
        self.assertEqual(rows[0]["all_assignees"], "甲公司|乙大学")
        self.assertEqual(rows[0]["application_number"], "")
        self.assertEqual(rows[0]["patent_type"], "发明申请")

    def test_rejects_broad_query_hit_without_exact_company_assignee(self):
        records = [{
            "publication_number": "CN123A",
            "title": "非目标公司的发明",
            "application_date": "2020-01-02",
            "assignees": "甲公司集团子公司 | 乙大学",
        }]
        fields = {key: key for key in ("publication_number", "title", "application_date", "assignees")}
        fields.update(application_number="application_number", patent_type="patent_type",
                      applicant_addresses="applicant_addresses")
        self.assertEqual(list(patent_rows(Company("1", "甲公司", ()), records, fields)), [])

    def test_normalizes_supported_assignee_shapes(self):
        self.assertEqual(normalize_assignees(["甲", {"original_name": "乙"}, {"value": "丙"}]), ["甲", "乙", "丙"])

    def test_splits_patsnap_pipe_delimited_assignees(self):
        self.assertEqual(normalize_assignees("甲公司 | 乙大学 | 甲公司"), ["甲公司", "乙大学"])

    def test_calibrated_p002_fields_and_address_override(self):
        fields = parse_field_mapping(["applicant_addresses=biblio.addresses"])
        self.assertEqual(fields["application_number"], "apno")
        self.assertEqual(fields["patent_type"], "patent_type")
        self.assertEqual(fields["applicant_addresses"], "biblio.addresses")

    def test_aligns_address_and_excludes_repaco_group_member(self):
        record = {
            "pn": "CN1U", "apno": "CN2020", "title": "装置", "apdt": 20200102,
            "original_assignee": "甲公司|甲子公司|乙大学",
            "address": "北京市|上海市|杭州市",
        }
        rows = list(patent_rows(Company("1", "甲公司", ()), [record], parse_field_mapping([]),
                                {"甲公司", "甲子公司"}))
        self.assertEqual([row["partner_name"] for row in rows], ["乙大学"])
        self.assertEqual(rows[0]["applicant_addresses"], "北京市|上海市|杭州市")
        self.assertEqual(rows[0]["listed_company_addresses"], "北京市")

    def test_repaco_reader_and_type_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repaco.csv"
            path.write_text("code,name\n1,甲公司（集团）\n", encoding="utf-8")
            self.assertEqual(read_repaco(path), {"1": {"甲公司(集团)"}})
        self.assertEqual(infer_patent_type("CN123S"), "外观设计")
        self.assertEqual(infer_patent_type("WO123A1"), "发明申请")


if __name__ == "__main__":
    unittest.main()
