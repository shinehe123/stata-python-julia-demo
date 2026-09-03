# stata-python-julia-demo
Codex环境测试用 -A cross-language research environment using Stata, Python, and Julia for Codex deployment.

## 中国上市公司专利合作数据

`patent_collaboration` 从**已获授权的智慧芽/OpenAPI 兼容检索接口**逐页读取中国专利，并把共同申请人展开成“上市公司—合作方—专利”明细。程序不会把 API 密钥写入源码、日志或输出；请勿提交 `.env`。

### 1. 准备公司表

复制 `data/companies.example.csv`，字段为：

* `code`：股票代码；
* `name`：用于匹配申请人的公司全称；
* `aliases`：可选，多个曾用名/简称用 `|` 分隔。

也可以把子公司逐行并入该文件：`code` 填母公司股票代码，`name` 填子公司全称。同一 `code` 下的所有名称会自动视为同集团成员，不会彼此产出合作关系；输出中的 `listed_company_name` 保留实际命中的母/子公司名称，便于追溯。

简称可能产生误匹配，生产任务建议使用智慧芽中的标准申请人名称或申请人 ID。合作专利定义为：检索结果中申请人至少两个，且除目标上市公司名称/别名和同集团成员外还存在其他申请人。

### 2. 字段映射、配置与运行

默认调用智慧芽中国区 P002 专利检索接口。通过环境变量传入密钥（不要把真实密钥写进源码或提交到 Git）：

```bash
export PATENT_API_KEY='请在本机填入密钥'
python -m patent_collaboration.cli \
  --companies data/companies.csv \
  --output data/output/patent_collaborations.csv
```

默认地址为 `https://connect.zhihuiya.com/search/patent/query-search-patent/v2`。实测 P002 搜索结果映射如下：

| 输出含义 | `--field` 名称 | P002 路径 | 说明 |
|---|---|---|---|
| 公告号 | `publication_number` | `pn` | 已校准 |
| 申请号 | `application_number` | `apno` | 已校准 |
| 申请日 | `application_date` | `apdt` | 已校准 |
| 申请人列表 | `assignees` | `original_assignee` | `|` 分隔 |
| 专利类型 | `patent_type` | `patent_type` | P002 未返回时按公告号末尾 A/B/U/S 推断 |
| 申请人地址 | `applicant_addresses` | `address` | 当前 P002 搜索响应未返回，输出留空；须用合同实际提供地址的接口/路径覆盖 |

P002 即使在请求体传入 `field` 也仍返回固定摘要字段。因此，工具的 `--field` 是**响应路径映射**，不是向 P002 申请额外字段。不能从摘要响应可靠获得“发明人地址”；此处按 CSMAR 口径输出的是申请人地址。地址列表与申请人列表等长时，工具按位置生成 `listed_company_addresses`。

其他合同版本可以映射嵌套路径，例如：

```bash
python -m patent_collaboration.cli \
  --companies data/companies.csv \
  --output data/output/patent_collaborations.csv \
  --results-path result.items --total-path result.total_count \
  --field publication_number=biblio.publication_number \
  --field application_number=biblio.application_number \
  --field patent_type=biblio.type \
  --field assignees=biblio.applicants \
  --field applicant_addresses=biblio.applicant_addresses
```

输出列按 CSMAR 常用口径包含公告号、申请号、申请日、类型和申请人列表，并保留合作方、标题、全部申请人地址及目标上市公司地址。缺少的可选字段写为空值，不会令整批任务失败。

### 3. 剔除同集团合作方（Repaco）

将 Repaco 企业关联数据整理成 UTF-8 CSV，至少包含 `code,name` 两列，其中 `code` 是母公司股票代码，`name` 是母公司或集团成员的精确全称：

```bash
python -m patent_collaboration.cli \
  --companies data/companies.csv \
  --repaco data/repaco_members.csv \
  --output data/output/patent_collaborations.csv
```

程序统一全角/半角括号并移除空白后做**精确名称匹配**，只剔除与当前股票代码同组的合作申请人，不使用容易误伤的包含匹配。建议保留 Repaco 原始版本日期和母子公司映射，以便研究结果复现。`data/repaco.example.csv` 展示输入格式。

### 4. 已运行的数据样例

仓库中的 `data/patent_collaborations.sample.csv` 是 2026-09-03 实际调用 P002 接口，并应用 `data/repaco.example.csv` 同集团剔除后得到的结果，范围为 `data/companies.example.csv` 中的贵州茅台（600519）和平安银行（000001）。样例共包含 12 条集团外合作专利明细；数据范围、接口和分公司计数记录在同名的 metadata JSON 中。平安银行样例命中的三条记录均只与同集团平安科技共同申请，故剔除后为零。该文件用于验证端到端流程，不代表全部中国上市公司。样例包含 P002 返回的申请号；P002 摘要未提供的地址为空，未伪造或从非授权来源补齐。

程序会再次检查每条结果的原始申请人中是否存在公司全称或配置的别名。因此，简称查询命中的集团子公司不会被误记为目标上市公司的合作专利。
