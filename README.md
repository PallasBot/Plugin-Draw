<p align="center">
  <img src="./assets/brand-avatar.png" width="220" height="220" alt="牛牛画画">
</p>

<h1 align="center">牛牛画画 pallas-plugin-draw</h1>

<p align="center">提供群内 AI 生图与参考图改图能力。</p>

<p align="center">
  <img alt="官方插件" src="https://img.shields.io/badge/%E5%AE%98%E6%96%B9%E6%8F%92%E4%BB%B6-FE7D37">
  <img alt="控制台插件商店" src="https://img.shields.io/badge/%E6%8E%A7%E5%88%B6%E5%8F%B0-%E6%8F%92%E4%BB%B6%E5%95%86%E5%BA%97-4EA94B">
  <img alt="安装命令" src="https://img.shields.io/badge/uv%20run%20pallas%20ext%20install%20pallas--plugin--draw-586069">
  <img alt="PyPI 版本" src="https://img.shields.io/pypi/v/pallas-plugin-draw?label=%E7%89%88%E6%9C%AC&color=2563EB">
</p>

## 安装方式

需已安装 [Pallas-Bot](https://github.com/PallasBot/Pallas-Bot) **≥ 4.0**。

推荐直接在控制台插件商店安装，或在本体项目中执行：

```bash
uv run pallas ext install pallas-plugin-draw
```

也可单独安装本包：

```bash
uv pip install pallas-plugin-draw
```

## 怎么使用

群内 AI 生图；可纯文字或带参考图（附图/回复图）。依赖画画网关，可配置每日次数。

可选安装社区插件 **牛牛爱发电（`afdian`）** 后，免费用尽可使用额外额度；未安装时行为与原先一致，不会出现付费相关提示。

### 用户命令

| 口令 / 触发 | 场景 | 说明 |
| --- | --- | --- |
| `牛牛画画 …` | 群内 | 按描述生图或改图 |

### 命令权限

| 命令 ID | 默认等级 |
| --- | --- |
| `draw.draw` | 所有人 |

> 详细用法、限制条件和可用范围以帮助为主。

## 配置项

WebUI **插件 → 牛牛画画** 或 **服务网关 / 连通性**；字段前缀为 `pallas_image_*`（历史命名，见文档站）。

完整键：本仓库 [`config.py`](src/pallas_plugin_draw/config.py)

## 排障

| 现象 | 处理 |
| --- | --- |
| 失败提示 | 看返回文案；用 `牛牛连通` 测网关 |
| 次数用尽 | 等待重置或调配额 |

## 实现

源码位置：[`src/pallas_plugin_draw/`](src/pallas_plugin_draw/)

关键文件：

- [`src/pallas_plugin_draw/__init__.py`](src/pallas_plugin_draw/__init__.py)：注册画画命令与插件元数据。
- [`src/pallas_plugin_draw/config.py`](src/pallas_plugin_draw/config.py)：定义画画网关与限制配置。

实现要点：

- 在 Bot 进程内直连配置的画画网关；主/备线可填 `provider_id` 沿用 AI · Provider 凭证，也可手动填写 base_url / api_key。
- 支持纯文本生成与带参考图的改图路径。
- 结果依赖外部画图网关，失败时以返回文案和连通性为主要事实源。

## 相关链接

| 说明 | 链接 |
| --- | --- |
| 牛牛画画 · 用户文档 | [文档站 · draw](https://PallasBot.github.io/Pallas-Bot-Docs/plugins/draw) |
| 插件开发入门 | [develop/plugin/getting-started](https://PallasBot.github.io/Pallas-Bot-Docs/develop/plugin/getting-started) |
