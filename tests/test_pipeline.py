from cnipa_collaboration.normalize import normalize_name, split_applicants
from cnipa_collaboration.pipeline import Company, build_tables, canonicalize


def test_normalize_and_split():
    assert normalize_name(" Ａ公司 （北京） ") == "a公司(北京)"
    assert split_applicants("甲公司；乙大学;甲公司") == ["甲公司", "乙大学"]


def test_alias_headers_and_collaboration_deduplication():
    company = Company("000001", "甲股份有限公司", ("甲公司",))
    row = {
        "申请（专利）号": "CN202010000001.2",
        "发明名称": "联合创新",
        "申请日": "2020-01-02",
        "申请（专利权）人": "甲公司；乙大学",
    }
    assert canonicalize(row)["application_number"] == "CN202010000001.2"
    patents, edges = build_tables([row, row], [company])
    assert len(patents) == 1
    assert patents[0]["is_collaboration"] == "true"
    assert edges == [{
        "company_id": "000001",
        "partner_name": "乙大学",
        "year": "2020",
        "joint_patent_count": 1,
    }]

