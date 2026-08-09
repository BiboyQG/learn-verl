import {cp, mkdir, rm} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(root, ".docusaurus-build");
const destination = path.join(root, "public/guide");

await rm(destination, {recursive: true, force: true});
await mkdir(path.dirname(destination), {recursive: true});
await cp(source, destination, {recursive: true});

console.log("Staged the Docusaurus build at public/guide/.");
