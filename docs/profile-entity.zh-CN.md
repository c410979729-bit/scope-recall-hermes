# Profile / Entity 只读视图

本文描述共享 Core 的读视图，以及 Hermes 和本地客户端 MCP 上的同名工具。

`profile` 与 `entity` 只读已承认的结构化 claim。空数据或尚未巩固的原始对话必须如实报告缺口，**不会**把原始聊天、USER.md、MEMORY.md 或共现文本自动写成档案。身份、路径与范围只来自已初始化的宿主上下文，工具参数不能传入。

`budget_tokens` 按**最终**规范化 JSON 的 UTF-8 字节计费（含 honesty 标记、身份字符串、provenance），**不是**分词器 token 数。默认 `max_items=16`、`budget_tokens=4096`。请求下限是 64，不会为了塞进空信封而抬高下限。若连最小的诚实空结果 / `unavailable` 信封都放不下，调用以 `ContractError(INPUT_INVALID, budget_tokens)` 拒绝；宿主错误包装**不是**视图结果。扫描封顶（候选并集在发布前裁到 200 对象栅栏内）时，`coverage` **绝不会**是 `complete_for_query`。`no_match` 只表示在当前范围与预算内没有可发布的当前项，**不是**全局不存在的证明。别名枚举触顶时不能证明唯一，返回未解析 / `ambiguous` 的部分缺口，不会按不完整前缀解析。

## 巩固前提

别名、目标事实和额外的别名证明对象共用发布上限。同一个对象只计一次；完整身份证明超过 200 个对象时，不截取部分证明来认定主体，而是返回扫描受限的空歧义响应。

必须先有捕获 + 已接受的巩固（`accept_claim_proposals` / 已承认的 claim）。只有资格为当前有效、且证据仍存活的 claim 才能进入视图。

- `proposed` / `superseded` / `retracted` / 抑制或删除：不是当前事实
- `disputed`：单独放在 `disputed`（profile）或用 `temporal_status=disputed` 标注（entity），不会当成已决
- 意图：仅 `pending` 进入 profile 的 `pending_intentions`；已完成 / 取消 / 过期不与当前工作混写
- 原始事件单独存在、没有任何承认的 claim：`status=no_match`，`gaps` 含 `consolidation_required`，正文不含聊天原文

## `profile`

对**显式写出的主体名**做确定性分类视图。默认当前已确认 / 已合格事实：事实、偏好、约束、决定；待处理意图单独成节。

### 请求（`profile_request`）

必填：`protocol_version`、`request_id`、`subject`、`max_items`（1–30）、`budget_tokens`（64–8000）。宿主工具可省略后两项，由适配器填默认值。禁止额外字段（含 `scope_id`、`project_id`、`data_directory`、`session_id`）。整数必须是 JSON integer，不能用 `true`/`1.0`。

### 当前这个人

`user`、`current_user`、`用户`、`我`、`me`、`myself`、`i` 表示当前这个人，英文不区分大小写；其他词按名称读取。此规则同时适用于 `profile` 和 `entity`。`me`、`myself`、`i` 是保留人称；库中即使存在字面上叫这些名字的主体，也不能用这些词按字面查询，没有当前身份时返回 `IDENTITY_UNBOUND`。

优先使用宿主已核实的人类发送者。本机共享入口的 Codex、Claude Code、WorkBuddy、dsh stdio MCP 在启动时显式启用主人回退：主体与该入口捕获用户输入时使用的主人身份相同，并重新核对清单是否批准这个身份。依据是安装时对这个入口本机主人的批准；程序不核对进程的系统账号是谁。谁能启动入口、读到配置和库，由操作系统的文件权限决定；共用账号或把入口目录开放给别的账号，等于把主人身份一起给了出去。这个回退不绑定对话，也不增加范围或写入权限。

通用上下文默认只有主人身份等价解析，不提供当前人回退；钩子、交接、私密状态检查和远程 HTTP 沿用这个默认。Hermes 的 CLI、Desktop、TUI、网关和 A2A 仍使用现有宿主说话人规则，未新增回退。没有共享清单批准或身份仍不明确时，人称查询返回 `IDENTITY_UNBOUND`，不按字面名字查询。

清单批准为同一主人的人类身份标记，在这两个视图中一起读取；已接受且证据有效的别名，其规范目标全部属于这组主人身份时，同样读取这些已批准的身份。多个规范目标在清单中属于同一主人时，`resolved_subject` 使用其中一个稳定的身份标记；每条事实仍保留自己的主体。存储主体和 slot key 保持原样，每条事实保留证据引用及可选的 `entries`（`{id, name}`，来自证据的入口）。清单未批准的身份不加入；没有已接受的身份链接时，主人的名字也只是一个名字。主体等价不授予读取权限，范围、项目、分支、发布检查、私密和暂停状态仍照常生效。

已验证别名范围之外的可读范围里存在这些名称的同名字面事实时，会阻止主人扩展：仍返回歧义，只按原查询名列出候选。

```json
{
  "protocol_version": "1.1",
  "request_id": "profile-1",
  "subject": "示例项目",
  "max_items": 16,
  "budget_tokens": 4096
}
```

### 响应要点

每条含 claim `ref`/`revision`、`temporal_status`、`evidence_refs`，以及证据上已承认的可选 `source_contexts`（`{platform, chat_type}`）和 `entries`。平台和入口只来自已存储证据，不来自本次查询文本。

```json
{
  "protocol_version": "1.1",
  "request_id": "profile-1",
  "status": "ok",
  "memory_epoch": 12,
  "subject": "示例项目",
  "resolved_subject": "示例项目",
  "alias_resolution": "literal",
  "sections": {
    "facts": [
      {
        "ref": "claim-…",
        "revision": 2,
        "kind": "fact",
        "subject": "示例项目",
        "predicate": "配色",
        "value_text": "蓝色",
        "conditions": [],
        "temporal_status": "current",
        "claim_state": "active",
        "valid_from": "2026-09-03T12:00:00.000000+00:00",
        "valid_to": null,
        "evidence_refs": ["event-…@1"],
        "basis": "direct_report",
        "source_contexts": [{"platform": "telegram", "chat_type": "private"}]
      }
    ],
    "preferences": [],
    "constraints": [],
    "decisions": [],
    "pending_intentions": []
  },
  "disputed": [],
  "gaps": [],
  "coverage": "complete_for_query",
  "answerability": "supported",
  "truncated": false,
  "scan_capped": false,
  "unmet_needs": []
}
```

`complete_for_query` 仅表示：**在当前允许的 scope / project / branch 内**，对这个显式主体的有界枚举没有触顶、也没有截断。它不是全库普查。

## `entity`

精确、证据接地的一跳视图。

| `action` | 含义 |
|----------|------|
| `probe` | 关于该主体的当前 **fact**（出边） |
| `related` | 直接记录的语句 / 关系 |

`direction`：`outgoing` | `incoming` | `both`。`probe` 固定作出边事实，忽略把方向理解成多跳。

- **出边**：库中记录的 `subject / predicate / value_text`
- **入边**：`value_text` **整段标量相等**于查询名（以及在别名已唯一解析时的规范主体）。不用共现、子串、逗号列表、自由文本枚举或改写来发明边
- 返回的 `value_text` 就是当时写下的宾语文本，**不**假装它有规范实体 ID
- `predicate` 若出现，必须完全相等
- 不递归、不画图；名称关系与主人等价只使用已接受的证据和清单批准

```json
{
  "protocol_version": "1.1",
  "request_id": "entity-1",
  "subject": "示例项目",
  "action": "related",
  "direction": "incoming",
  "max_items": 16,
  "budget_tokens": 4096
}
```

入边命中示例（整段相等）：

```json
{
  "direction": "incoming",
  "subject": "小林",
  "predicate": "负责",
  "value_text": "示例项目",
  "ref": "claim-…",
  "revision": 1,
  "kind": "fact",
  "conditions": ["未授权"],
  "temporal_status": "current",
  "claim_state": "active",
  "valid_from": "2026-09-01T12:00:00.000000+00:00",
  "valid_to": null,
  "evidence_refs": ["event-…@1"],
  "basis": "direct_report"
}
```

有条件的直接关系会保留 `conditions` 与 `valid_from` / `valid_to`，不会被渲染成无条件边。若记录的是 `value_text: "Bob, Carol"`，查询 `Bob` **不会**得到入边。入边枚举匹配**仍保留的版本**上的标量，再由 `select_effective` 与整段相等复检决定是否当前；过期 / 被替代的旧值不会当成当前。

## 已验证的名称关系

读取只使用已生效的 `kind=alias` 记录。开启范围的发布前审查后，自动身份关系也须先批准。明确项目改名需要名称身份事实和完整改名证据；
模型判断的实体关系需要保存的身份判断证据，并绑定两端记录的具体版本。
两类关系都重验来源、范围、项目、分支、有效期、撤回和删除状态，不在读取时推测身份。

| 情况 | 结果 |
| --- | --- |
| 唯一且证据有效的关系 | `alias_resolution=resolved`，读取规范主体及其已验证的等价名称 |
| 同名对应不同主体或不同范围 | `alias_resolution=ambiguous`，不任选一个主体；可发布的原名称事实按范围放在可选 `candidate_groups` |
| 枚举触及候选上限 | 返回 `ambiguous` 和部分覆盖缺口 |
| 关系未生效、已撤回或两端证据失效 | 按原名称查询 |

### 同名候选分组

歧义响应的旧顶层字段沿用原有语义：保留 `status=partial`、`answerability=ambiguous`、`resolved_subject=null` 和 `alias_ambiguous`，顶层 `sections` / `disputed` 或 `statements` 仍为空。`gaps` 可能增加预算、条数、密钥或争议缺口，发布检查失败仍可返回 `unavailable`。可选的 `candidate_groups` 提供候选内容，不表示不同组是同一个主体：

```json
{
  "candidate_groups": [
    {
      "audience": {"scope_id": "示例项目范围", "project_id": null, "branch_id": null},
      "sections": {"facts": [], "preferences": [], "constraints": [], "decisions": [], "pending_intentions": []},
      "disputed": []
    }
  ]
}
```

上面展示字段形状；实际响应只保留有可发布条目的组。`entity` 的每组使用同一个 `audience`，正文为 `statements` 数组。范围标识来自已发布事实的 `scope_id`、`project_id` 和 `branch_id`，调用方无权读取的范围不会出现在分组、计数或歧义判断中。名称碰撞检查保持保守；发布时仍执行证据的项目和分支精确匹配，不能因分组扩大读取。别名歧义时只列原查询名下的事实，不追随有争议的关系。

所有组与顶层正文共用 `max_items` 和最终 JSON 字节预算，来源及范围标识也计入预算。预算裁剪后空组会被移除，候选扫描仍共用上限；别名枚举触顶时不输出基于不完整身份关系的候选。不同组不合并、不补全，也不推断身份。`memory_epoch` 保持全库变更计数的既有含义。

候选扩展一条也放不下预算时，退回原来的空歧义响应，不带 `candidate_groups`，也不带因候选扩展新增的缺口或提示；原空响应放得下的预算仍可返回。能放下一部分时才保留分组及相应的截断标记。

## 部分结果与只读

- `truncated` / `max_items_cap` / `budget_token_cap` / `scan_capped`：有界、如实的部分结果；没有续读游标
- 选择与发布之间数据变化：失败关闭，返回 `unavailable`，不带过期档案 / 边 / 别名正文
- 清单复核只保证到最后一次身份解析，不覆盖此后的清单变化，也不提供清单与数据库之间的原子快照
- 视图查询不修改事实、不写审计、不入队、不碰向量；不需要 embedding / LLM 配置。访问旧结构前仍可能触发协调升级
- Hermes 与本地客户端共用同一套 Core 方法；适配器只做身份绑定与参数边界

机器可读 schema：`contracts/profile_request.schema.json`、`contracts/entity_request.schema.json`、`contracts/profile_view.schema.json`、`contracts/entity_view.schema.json`。


## 实体名称与合并

提炼可以给出规范实体名，并在 `entity_type` 和 `entity_names` 中保留类型与证据里的原称。
旧记录没有这些可选字段时，类型按 `unknown`、原称按空列表处理。实体范围包括人、动物和物品、地点、组织及项目。
同名人物应使用原文支持的职业、所属组织或地点消歧；仅有名字相同或拼写相似不足以合并。

后台 worker 先通过现有向量搜索和字面相似度查找候选，再通过已配置的辅助模型判断身份。
模型只可选择已提供的记录版本；不确定或不可用时不合并。模型请求受同一预算账本和私密模式约束。
结果保存为带原文证据的别名记录，两端原事实保持原样。范围、项目和分支必须完全相同。
已有的明确项目改名仍然可用。
判断结果按范围内的名字保存，后续事实中的主语和原称均已判断时不再请求模型。
端点改版或删除使对应名字重新进入有界回填；主人明确撤回的关系保持撤回。

`entity` 同时读取已合并名称下的一跳记录；`trace` 沿这些实体的已记录关系行走，条件和跳数限制照常。路径上的每条边保留可选的 `entries` 来源入口标签；没有入口标签时不增加这个字段。
召回只把问题中的名称作为同一实体的等价称呼匹配，不额外沿实体关系扩展。
同一位置出现多个已知名称时，优先匹配完整的较长名称，避免短别名改写另一个实体的名称。
撤回别名记录后，查询按原名称工作；两端记录或证据失效时，该别名也停止生效。

首次打开旧库先完成存储升级，实体解析由 worker 分批处理。回填完成前，原名称的查询仍可使用。
Doctor 的 `entity_resolution` 检查显示已处理和仍待处理的记录数。需要 worker 提供身份判断能力；
该能力不可用时进度保留，已有记忆照常按原名称查询。
