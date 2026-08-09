import type {Config} from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

const commit = "d33ddd7140f44d392e0e10b48a8902651a1340f4";

const config: Config = {
  title: "verl 深度学习手册",
  tagline: "从 RL 基础到 V1 源码数据流",
  favicon: "img/verl-logo.png",

  url: "https://learn-verl.banghaochi.com",
  baseUrl: "/guide/",
  trailingSlash: true,

  organizationName: "banghaochi",
  projectName: "learn-verl",

  onBrokenLinks: "throw",
  onBrokenAnchors: "throw",
  markdown: {
    format: "detect",
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: "throw",
      onBrokenMarkdownImages: "throw",
    },
  },

  i18n: {
    defaultLocale: "zh-CN",
    locales: ["zh-CN"],
    localeConfigs: {
      "zh-CN": {
        htmlLang: "zh-CN",
        label: "简体中文",
      },
    },
  },

  presets: [
    [
      "classic",
      {
        docs: {
          path: "content",
          routeBasePath: "/",
          numberPrefixParser: false,
          sidebarPath: "./sidebars.ts",
          breadcrumbs: true,
          showLastUpdateTime: false,
          showLastUpdateAuthor: false,
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        pages: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: [
    "@docusaurus/theme-mermaid",
    [
      "@easyops-cn/docusaurus-search-local",
      {
        hashed: true,
        language: ["en", "zh"],
        docsDir: "content",
        docsRouteBasePath: "/",
        indexDocs: true,
        indexBlog: false,
        indexPages: false,
        highlightSearchTermsOnTargetPage: true,
        searchResultLimits: 12,
        searchResultContextMaxLength: 70,
        zhUserDict: [
          "verl",
          "AgentLoop",
          "ToolAgentLoop",
          "TransferQueue",
          "TensorDict",
          "DataProto",
          "KVBatchMeta",
          "rollout",
          "reward",
          "advantage",
          "GRPO",
          "PPO",
          "FSDP",
          "Megatron",
          "vLLM",
          "SGLang",
        ].join("\n"),
      },
    ],
  ],

  themeConfig: {
    image: "img/og.png",
    metadata: [
      {name: "keywords", content: "verl, RL, GRPO, PPO, Agent Loop, Tool Agent, 强化学习, 大模型后训练"},
      {name: "author", content: "banghaochi"},
    ],
    announcementBar: {
      id: "snapshot",
      content: `非官方中文学习手册 · 对应 verl 0.9.0.dev · 源码快照 ${commit.slice(0, 8)}`,
      backgroundColor: "#0f766e",
      textColor: "#f0fdfa",
      isCloseable: false,
    },
    navbar: {
      title: "verl 深度学习手册",
      hideOnScroll: true,
      logo: {
        alt: "verl logo",
        src: "img/verl-logo.png",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "教程",
        },
        {
          to: "/15_source_map_and_glossary/",
          label: "源码地图",
          position: "left",
        },
        {
          href: `https://github.com/verl-project/verl/tree/${commit}`,
          label: "对应源码",
          position: "right",
        },
        {
          href: "https://github.com/BiboyQG/learn-verl",
          label: "本站源码",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "开始学习",
          items: [
            {label: "学习地图", to: "/00_learning_map/"},
            {label: "必要前置知识", to: "/01_prerequisites/"},
            {label: "端到端数据流", to: "/12_end_to_end_training_flow/"},
          ],
        },
        {
          title: "Agent 专题",
          items: [
            {label: "Agent Loop", to: "/08_agent_loop/"},
            {label: "Tool Agent Loop", to: "/09_tool_agent_loop/"},
          ],
        },
        {
          title: "项目",
          items: [
            {label: "本站源码", href: "https://github.com/BiboyQG/learn-verl"},
            {label: "verl GitHub", href: "https://github.com/verl-project/verl"},
            {label: "verl 官方文档", href: "https://verl.readthedocs.io/"},
          ],
        },
      ],
      copyright:
        "非官方学习资料。verl 名称与项目归其原作者和贡献者所有。",
    },
    colorMode: {
      defaultMode: "light",
      respectPrefersColorScheme: true,
    },
    prism: {
      additionalLanguages: ["python", "yaml", "bash", "json", "markdown"],
    },
    mermaid: {
      theme: {light: "neutral", dark: "dark"},
      options: {
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif',
      },
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
