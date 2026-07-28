# 更新日志

本文件依据 git tag 历史整理，版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。
新提交合入后请在 `## [Unreleased]` 下记录，发布时随版本 tag 归档。

## [Unreleased]

## [4.1.10] - 2026-07-28

- fix(draw): 日用量读写更稳：废弃 `total_entries` 格式告警、持久化失败不再静默；仍用短 id `draw` 对接 Bot storage 别名

## [4.1.9] - 2026-07-27

- fix(draw): 分片画画统计回灌与集群合并，避免重启后用量被偏少快照盖掉

## [4.1.8] - 2026-07-26

- feat(llm_tools): 为口令工具补充口语 hints

## [4.1.7] - 2026-07-26

- fix: 日用量 storage 延后到启动后再读，避免 import 阶段 `undeclared plugin_storage key`

## [4.1.6] - 2026-07-26

- feat(config): WebUI 配置字段增加 ui_group 分组与 ui_order 排序


## [4.1.5] - 2026-07-25

- feat(config): `pallas_image_api_backends` 声明 `ui_provider_gateway`（split 主备线路面板）

## [4.1.4] - 2026-07-25

- feat: PluginMetadata.extra 增加 `help_tag`（帮助图分组）

## [4.1.2] - 2026-07-25

- feat(draw): 可选软接入社区插件 `afdian`；未安装/未启用时仅免费日限，拒画文案不含付费引导
- feat(draw): 接入后免费用尽可扣额外额度；成功后互斥扣次
- fix(draw): `run_backend_param_attempts` / `finish_draw_failure` 接受 keyword-only `paid_afdian`、`at_user_id`

## [4.1.1] - 2026-07-24

- feat(stats): 运维向日统计（成功/失败、按网关/Provider/模型）与可选费用累计
- feat(config): 主/备线 `cost_per_image` 与全局 `pallas_image_stats_cost_currency`

## [4.1.0] - 2026-07-24

- feat(config): 主/备线可填 `provider_id`，运行时沿用 AI · Provider 的 base_url 与 api_key
- breaking(config): 移除 `pallas_image_runtime_mode` 与 AI Runtime 回退/熔断项；画画仅插件直连网关
- docs(readme): 实现说明与仅直连网关对齐

## [4.0.17] - 2026-07-20

- feat(config): 默认 `plugin_runtime` 直连画画网关；`ai_service_runtime` 降为兼容旧路径
- docs: README 实现说明与默认模式对齐

## [4.0.16] - 2026-07-13

- feat(draw): `ai_service_runtime` 请求携带 Bot 画画网关（主+备线）供 AI 按序执行

## [4.0.15] - 2026-07-13

- fix(draw): 画画消息去重签名纳入时间戳，同描述可多次生图

## [4.0.14] - 2026-07-10
- chore(draw): 移除冗余「牛牛网关」命令声明（改由内核牛牛连通提供）
- feat(config): `pallas_image_runtime_mode` 改为枚举，WebUI 下拉选择

## [4.0.13] - 2026-07-10
- fix(draw): 传输超时/断连优先切备线，避免主网关参数重试耗尽总超时
- fix(draw): 有备线时主网关单次超时预留后续预算

## [4.0.12] - 2026-06-27
- docs(readme): 命令权限默认等级改用中文展示

## [4.0.11] - 2026-06-27
- docs(readme): 「怎么使用」口令统一加行内代码标记

## [4.0.10] - 2026-06-27
- fix(gateway_probe): 插件直连模式不展示 AI runtime 状态行

## [4.0.9] - 2026-06-25
- feat(draw): 备用网关单次超时与请求预算均分

## [4.0.8] - 2026-06-25
- refactor(draw): 熔断状态改读 AI /health 缓存
- feat(metadata): 补充画画网关命令冷却声明

## [4.0.7] - 2026-06-24
- feat(knowledge): 声明 knowledge_sources FAQ 供 LLM 注入

## [4.0.6] - 2026-06-19
- docs(assets): 更新头像资源并改用 PyPI 版本徽章
- chore(assets): 替换品牌头像为透明背景版本

## [4.0.5] - 2026-06-18
- docs(readme): 统一官方插件卡片模板

## [4.0.4] - 2026-06-18
- docs(readme): 更新官方扩展安装命令

## [4.0.3] - 2026-06-18
- migrate: src.* → pallas.api.* / pallas.product.* / pallas.core.*
- release: bump to 4.0.3 for pallas import migration

## [4.0.2] - 2026-06-18
- docs(readme): 添加 Pallas-Bot hero 图
- feat(draw): AI 服务双栈模式与插件层 runtime 止血
- feat(draw): 对齐 bundled draw 的 media task callback 慢路径
- feat(draw): 4.0.2 瘦身并承接 AI runtime 主路径

## [4.0.1] - 2026-06-17
- feat: Pallas-Bot 4.0 官方扩展首包
- fix(build): 修正 hatch wheel 的 src 包路径
- feat(release): PyPI 发版 workflow 与 4.0.1
