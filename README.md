# stata-python-julia-demo
Codex环境测试用 -A cross-language research environment using Stata, Python, and Julia for Codex deployment.

## 中国上市公司专利合作数据

`patent_collaboration` 从**已获授权的智慧芽/OpenAPI 兼容检索接口**逐页读取中国专利，并把共同申请人展开成“上市公司—合作方—专利”明细。程序不会把 API 密钥写入源码、日志或输出；请勿提交 `.env`。

### 1. 准备公司表

复制 `data/companies.example.csv`，字段为：

* `code`：股票代码；
* `name`：用于匹配申请人的公司全称；
* `aliases`：可选，多个曾用名/简称用 `|` 分隔。

简称可能产生误匹配，生产任务建议使用智慧芽中的标准申请人名称或申请人 ID。合作专利定义为：检索结果中申请人至少两个，且除目标上市公司名称/别名外还存在其他申请人。

### 2. 配置并运行

默认调用智慧芽中国区 P002 专利检索接口。通过环境变量传入密钥（不要把真实密钥写进源码或提交到 Git）：

```bash
export PATENT_API_KEY='请在本机填入密钥'
python -m patent_collaboration.cli \
  --companies data/companies.csv \
  --output data/output/patent_collaborations.csv
```

默认地址为 `https://connect.zhihuiya.com/search/patent/query-search-patent/v2`，请求使用 `query_text`、`offset` 和 `limit`，读取 `data.results` 和 `data.total_search_result_count`，并使用 P002 的 `pn`、`title`、`apdt`、`original_assignee` 字段。如果合同使用国际区或其他版本，可通过参数适配，例如：

```bash
python -m patent_collaboration.cli \
  --companies data/companies.csv \
  --output data/output/patent_collaborations.csv \
  --results-path result.items --total-path result.total_count \
  --field publication_number=biblio.publication_number \
  --field assignees=biblio.applicants
```

运行 `python -m patent_collaboration.cli --help` 查看查询参数名、查询模板、重试和限速等选项。请仅使用合同允许的 API，并遵守智慧芽的服务条款、频率限制和数据许可；本工具不绕过登录、验证码或访问控制。

### 3. 已运行的数据样例

仓库中的 `data/patent_collaborations.sample.csv` 是 2026-09-03 实际调用 P002 接口得到的结果，范围为 `data/companies.example.csv` 中的贵州茅台（600519）和平安银行（000001）。样例共包含 15 条合作专利明细；数据范围、接口和分公司计数记录在同名的 metadata JSON 中。该文件用于验证端到端流程，不代表全部中国上市公司。

程序会再次检查每条结果的原始申请人中是否存在公司全称或配置的别名。因此，简称查询命中的集团子公司不会被误记为目标上市公司的合作专利。
