import {access, readFile, readdir} from "node:fs/promises";
import {JSDOM} from "jsdom";
import path from "node:path";
import {fileURLToPath} from "node:url";

const mermaidDom = new JSDOM("<!doctype html><html><body></body></html>");
globalThis.window = mermaidDom.window;
globalThis.document = mermaidDom.window.document;
const {default: mermaid} = await import("mermaid");

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const build = path.join(root, ".docusaurus-build");
const content = path.join(root, "docs-site/content");
const commit = "d33ddd7140f44d392e0e10b48a8902651a1340f4";
const routes = [
  "",
  "00_learning_map",
  "01_prerequisites",
  "02_architecture",
  "03_configuration_and_entrypoint",
  "04_data_and_protocols",
  "05_ray_controller_and_workers",
  "06_model_engines_and_parallelism",
  "07_rollout_and_weight_sync",
  "08_agent_loop",
  "09_tool_agent_loop",
  "10_reward_and_advantage",
  "11_policy_and_value_update",
  "12_end_to_end_training_flow",
  "13_training_modes_and_checkpoints",
  "14_extension_and_debugging",
  "15_source_map_and_glossary",
];

const markdownFiles = (await readdir(content)).filter((name) => name.endsWith(".md"));
let parsedMermaidCount = 0;
for (const name of markdownFiles) {
  const markdown = await readFile(path.join(content, name), "utf8");
  const diagrams = [...markdown.matchAll(/^```mermaid\s*\n([\s\S]*?)^```\s*$/gm)];
  for (const [index, diagram] of diagrams.entries()) {
    try {
      await mermaid.parse(diagram[1]);
    } catch (error) {
      throw new Error(`Invalid Mermaid diagram ${index + 1} in ${name}.`, {cause: error});
    }
    parsedMermaidCount += 1;
  }
}
if (parsedMermaidCount !== 37) {
  throw new Error(`Expected 37 Mermaid diagrams, found ${parsedMermaidCount}.`);
}

const pages = [];
for (const route of routes) {
  const file = route ? path.join(build, route, "index.html") : path.join(build, "index.html");
  await access(file);
  pages.push(await readFile(file, "utf8"));
}

const html = pages.join("\n");
const sourceLinkPrefix = `https://github.com/verl-project/verl/blob/${commit}/`;
const sourceTreePrefix = `https://github.com/verl-project/verl/tree/${commit}/`;
const sourceLinkCount =
  html.split(sourceLinkPrefix).length - 1 + html.split(sourceTreePrefix).length - 1;
const katexCount = (html.match(/class="katex/g) ?? []).length;
const katexErrorCount = (html.match(/class="katex-error/g) ?? []).length;
const javascriptDirectory = path.join(build, "assets/js");
const javascriptFiles = (await readdir(javascriptDirectory)).filter((name) => name.endsWith(".js"));
const javascript = (
  await Promise.all(
    javascriptFiles.map((name) => readFile(path.join(javascriptDirectory, name), "utf8")),
  )
).join("\n");
const mermaidCount = (javascript.match(/\.mermaid,\{value:/g) ?? []).length;

if (sourceLinkCount < 850) {
  throw new Error(`Expected at least 850 pinned source links, found ${sourceLinkCount}.`);
}
if (katexCount < 50) {
  throw new Error(`Expected at least 50 KaTeX nodes, found ${katexCount}.`);
}
if (katexErrorCount > 0) {
  throw new Error(`Found ${katexErrorCount} KaTeX rendering errors.`);
}
if (mermaidCount < 37) {
  throw new Error(`Expected at least 37 Mermaid markers, found ${mermaidCount}.`);
}
if (html.includes('href="../../')) {
  throw new Error("Found unresolved repository-relative links in generated HTML.");
}

const rootFiles = await readdir(build);
const hasSearchIndex = rootFiles.some((name) => name.includes("search-index"));
if (!hasSearchIndex) {
  throw new Error("Could not find the generated local-search assets.");
}

console.log(
  `Validated ${routes.length} routes, ${sourceLinkCount} pinned source links, ` +
    `${katexCount} KaTeX nodes, and ${parsedMermaidCount} parsed Mermaid diagrams.`,
);
