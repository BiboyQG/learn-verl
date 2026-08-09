import type {SidebarsConfig} from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {type: "doc", id: "index", label: "手册首页"},
    {
      type: "category",
      label: "第一部分：建立心智模型",
      collapsed: false,
      items: [
        {type: "doc", id: "00_learning_map", label: "00 · 学习地图"},
        {type: "doc", id: "01_prerequisites", label: "01 · 必要前置知识"},
        {type: "doc", id: "02_architecture", label: "02 · 整体架构"},
      ],
    },
    {
      type: "category",
      label: "第二部分：框架运行机制",
      collapsed: false,
      items: [
        {type: "doc", id: "03_configuration_and_entrypoint", label: "03 · 配置与入口"},
        {type: "doc", id: "04_data_and_protocols", label: "04 · 数据与协议"},
        {type: "doc", id: "05_ray_controller_and_workers", label: "05 · Ray 与 Worker"},
        {type: "doc", id: "06_model_engines_and_parallelism", label: "06 · 模型引擎与并行"},
        {type: "doc", id: "07_rollout_and_weight_sync", label: "07 · Rollout 与权重同步"},
      ],
    },
    {
      type: "category",
      label: "第三部分：Agent 与工具",
      collapsed: false,
      items: [
        {type: "doc", id: "08_agent_loop", label: "08 · Agent Loop"},
        {type: "doc", id: "09_tool_agent_loop", label: "09 · Tool Agent Loop"},
      ],
    },
    {
      type: "category",
      label: "第四部分：RL 算法与更新",
      collapsed: false,
      items: [
        {type: "doc", id: "10_reward_and_advantage", label: "10 · Reward 与 Advantage"},
        {type: "doc", id: "11_policy_and_value_update", label: "11 · Policy / Value 更新"},
        {type: "doc", id: "12_end_to_end_training_flow", label: "12 · 端到端训练流"},
      ],
    },
    {
      type: "category",
      label: "第五部分：运行、扩展与调试",
      collapsed: false,
      items: [
        {type: "doc", id: "13_training_modes_and_checkpoints", label: "13 · 训练模式与 Checkpoint"},
        {type: "doc", id: "14_extension_and_debugging", label: "14 · 扩展与调试"},
        {type: "doc", id: "15_source_map_and_glossary", label: "15 · 源码地图与术语表"},
      ],
    },
  ],
};

export default sidebars;
