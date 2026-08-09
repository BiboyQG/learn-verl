import assert from "node:assert/strict";
import {access, readFile} from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("packages the complete Docusaurus guide into the Sites build", async () => {
  const guide = new URL("dist/client/guide/", root);
  const routes = [
    "index.html",
    "00_learning_map/index.html",
    "09_tool_agent_loop/index.html",
    "12_end_to_end_training_flow/index.html",
    "15_source_map_and_glossary/index.html",
  ];

  await Promise.all(routes.map((route) => access(new URL(route, guide))));

  const home = await readFile(new URL("index.html", guide), "utf8");
  assert.match(home, /verl 深度学习手册/);
  assert.match(home, /非官方中文学习手册/);
  assert.match(home, /search/i);
});

test("keeps the Sites worker entrypoint and root guide redirect", async () => {
  await access(new URL("dist/server/index.js", root));

  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  const layout = await readFile(new URL("app/layout.tsx", root), "utf8");
  assert.match(page, /redirect\("\/guide\/"\)/);
  assert.match(layout, /learn-verl\.banghaochi\.com/);
  assert.doesNotMatch(page + layout, /codex-preview|SkeletonPreview/);
});
