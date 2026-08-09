# learn-verl

[verl 深度学习手册](https://learn-verl.banghaochi.com/guide/)的站点源码。这是一套非官方中文教程，按“强化学习基础 → verl 架构 → V1 运行时 → Agent Loop → 算法与扩展”的顺序，系统讲解 verl 的 RL post-training 框架。

> 教程对应 `verl` `0.9.0.dev`，源码快照固定为 [`d33ddd71`](https://github.com/verl-project/verl/tree/d33ddd7140f44d392e0e10b48a8902651a1340f4)。固定快照可避免源码行号链接随 `main` 漂移。

## 内容与能力

- 17 个章节，包含从入门路线到源码地图与术语表的完整学习路径
- 571 个代码块、37 张 Mermaid 图，以及 KaTeX 数学公式
- 中文本地全文搜索、暗色模式、移动端阅读和上一章/下一章导航
- 907 个指向固定 verl commit 的源码链接
- 构建时检查全部章节路由、源码链接、公式、Mermaid 与搜索索引

教程正文位于 [`docs-site/content`](docs-site/content)，站点使用 Docusaurus 3；根路由和 Sites 部署包装位于 `app`、`worker` 与 `vite.config.ts`。

## 本地开发

需要 Node.js `>=22.13.0`。

```bash
npm ci
npm run dev
```

本地入口为 `http://localhost:3000/`，会自动跳转到 `/guide/`。

提交前运行：

```bash
npm run lint
npm test
```

`npm test` 会完成 Docusaurus 静态构建、内容完整性校验、Sites/Vinext 构建和产物测试。

## 修改教程

公开仓库中可直接编辑 `docs-site/content/*.md`。写作约定：

- 行内公式使用 `$...$`，块级公式使用 `$$...$$`
- 图使用 fenced code block：<code>```mermaid</code>
- 代码块标注语言，例如 `python`、`yaml`、`bash`、`json`
- 指向 verl 源码的链接使用固定 commit permalink，不使用浮动的 `main`
- 不要编辑 `.docusaurus-build`、`public/guide` 或 `dist`，它们都是生成产物

若本仓库位于 verl checkout 的 `website/learn-verl` 下，可从原始教程目录重新同步：

```bash
npm run sync:docs
```

也可显式传入其他教程目录：

```bash
npm run sync:docs -- /absolute/path/to/learn_verl_zh
```

## 部署结构

1. Docusaurus 将教程构建到 `.docusaurus-build`。
2. 校验脚本检查 17 个路由、固定源码链接、KaTeX、Mermaid 与搜索索引。
3. 静态文件被复制到 `public/guide`。
4. Vinext 生成 Sites 可部署产物，并将站点根路径重定向到 `/guide/`。

## 声明与许可

本项目是非官方学习资料，不隶属于 verl 项目。`verl` 名称、标识和上游源码归其原作者与贡献者所有。

本站源码与教程按 [Apache License 2.0](LICENSE) 发布。项目在 AI 辅助下创建和维护；所有技术内容应以固定快照对应的源码为准。
