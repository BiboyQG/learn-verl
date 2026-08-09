import {mkdir, readFile, readdir, rm, writeFile} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.resolve(root, "../../docs/learn_verl_zh");
const destination = path.join(root, "docs-site/content");
const commit = "d33ddd7140f44d392e0e10b48a8902651a1340f4";
const repositoryRoot = "docs/learn_verl_zh";
const directoryTargets = new Set(["tests/checkpoint_engine"]);

function rewriteRepositoryLinks(markdown) {
  let fenceCharacter = null;

  return markdown
    .split("\n")
    .map((line) => {
      const fence = line.match(/^\s*(`{3,}|~{3,})/);
      if (fence) {
        const character = fence[1][0];
        fenceCharacter = fenceCharacter === null ? character : fenceCharacter === character ? null : fenceCharacter;
        return line;
      }
      if (fenceCharacter !== null) return line;

      return line.replace(/\]\((\.\.\/[^)]+)\)/g, (full, url) => {
        const match = url.match(/^([^?#]+)(\?[^#]*)?(#.*)?$/);
        if (!match) return full;

        const [, relativePath, query = "", fragment = ""] = match;
        const repositoryPath = path.posix.normalize(
          path.posix.join(repositoryRoot, decodeURIComponent(relativePath)),
        );
        if (repositoryPath.startsWith(`${repositoryRoot}/`)) return full;

        const objectType = directoryTargets.has(repositoryPath) ? "tree" : "blob";
        return `](https://github.com/verl-project/verl/${objectType}/${commit}/${repositoryPath}${query}${fragment})`;
      });
    })
    .join("\n");
}

await rm(destination, {recursive: true, force: true});
await mkdir(destination, {recursive: true});

const markdownFiles = (await readdir(source))
  .filter((name) => name.endsWith(".md"))
  .sort();

for (const name of markdownFiles) {
  const outputName = name === "README.md" ? "index.md" : name;
  const markdown = await readFile(path.join(source, name), "utf8");
  await writeFile(
    path.join(destination, outputName),
    rewriteRepositoryLinks(markdown),
    "utf8",
  );
}

console.log(`Synced ${markdownFiles.length} Markdown files into docs-site/content/.`);
