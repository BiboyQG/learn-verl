# Agent instructions

These rules apply to the entire repository.

## Purpose

- Maintain the non-official Chinese `verl` learning site.
- Keep explanations grounded in the pinned upstream source snapshot.
- Optimize for accurate, readable documentation rather than framework novelty.

## Source layout

- `docs-site/content/`: canonical public tutorial Markdown.
- `docs-site/docusaurus.config.ts`: docs routing, search, Mermaid, KaTeX, and site metadata.
- `docs-site/sidebars.ts`: deliberate chapter order.
- `docs-site/src/css/custom.css`: documentation-specific presentation.
- `app/` and `worker/`: thin Sites/Vinext wrapper and `/guide/` redirect.
- `scripts/`: source synchronization, staging, and output validation.
- `tests/`: deployable-output tests.

Do not edit generated directories: `.docusaurus-build/`, `public/guide/`, `dist/`, `.docusaurus/`, or `docs-site/.docusaurus/`.

## Documentation rules

- Preserve the learning progression from high-level concepts to low-level runtime details.
- Treat `d33ddd7140f44d392e0e10b48a8902651a1340f4` as the pinned `verl` snapshot until an explicit upgrade updates the whole site.
- Link source claims to GitHub permalinks at the pinned commit. Do not use line-number links against `main`.
- Use `$...$` for inline math and `$$...$$` for display math.
- Use fenced `mermaid` blocks for diagrams and language-tagged fences for code.
- Keep one H1 per chapter and retain the chapter filename/order unless navigation is intentionally redesigned.
- State clearly that the site is non-official.
- Do not invent APIs, configuration keys, defaults, shapes, or control flow. Verify technical changes against upstream source.

## Site changes

- Keep the guide functional in light and dark modes and on narrow screens.
- Preserve local Chinese search, code highlighting, Mermaid, KaTeX, heading anchors, and previous/next navigation.
- Do not add client analytics, external fonts, trackers, authentication, or secrets without explicit approval.
- Prefer existing Docusaurus and Sites mechanisms over custom runtime code.

## Verification

Run these commands before handing off changes:

```bash
npm run lint
npm test
```

`npm test` must build both layers and validate all 17 routes, pinned source links, KaTeX output, Mermaid markers, the search index, the Sites worker, and the root redirect.

If only synchronizing content from a local `verl` checkout, run `npm run sync:docs` first and inspect the Markdown diff before testing.

## Change discipline

- Keep diffs scoped and remove superseded starter or generated code.
- Never commit credentials, deployment write tokens, `.env*`, or generated artifacts.
- Update `README.md` when commands, layout, deployment, domain, or the pinned source snapshot changes.
- Human review is required for technical claims and every changed line before publication.
