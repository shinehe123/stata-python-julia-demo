import csv
import tempfile
import unittest
from pathlib import Path

from patent_collaboration.cli import Company, normalize_assignees, patent_rows, query_for, read_companies


class CollaborationTests(unittest.TestCase):
    def test_reads_utf8_bom_and_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "companies.csv"
            path.write_text("\ufeffcode,name,aliases\n1,甲公司,甲|甲股份\n", encoding="utf-8")
            self.assertEqual(read_companies(path), [Company("1", "甲公司", ("甲", "甲股份"))])

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
        rows = list(patent_rows(Company("1", "甲公司", ()), records, fields))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["partner_name"], "乙大学")
        self.assertEqual(rows[0]["all_assignees"], "甲公司|乙大学")

    def test_rejects_broad_query_hit_without_exact_company_assignee(self):
        records = [{
            "publication_number": "CN123A",
            "title": "非目标公司的发明",
            "application_date": "2020-01-02",
            "assignees": "甲公司集团子公司 | 乙大学",
        }]
        fields = {key: key for key in ("publication_number", "title", "application_date", "assignees")}
        self.assertEqual(list(patent_rows(Company("1", "甲公司", ()), records, fields)), [])

    def test_normalizes_supported_assignee_shapes(self):
        self.assertEqual(normalize_assignees(["甲", {"original_name": "乙"}, {"value": "丙"}]), ["甲", "乙", "丙"])

    def test_splits_patsnap_pipe_delimited_assignees(self):
        self.assertEqual(normalize_assignees("甲公司 | 乙大学 | 甲公司"), ["甲公司", "乙大学"])


if __name__ == "__main__":
    unittest.main()
