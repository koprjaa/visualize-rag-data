# chromadb-embedding-visualizer

**Fly around your ChromaDB vector store in 3D — UMAP + HDBSCAN in the backend, Three.js in the browser, instant load from pre-computed coordinates.**

![nextjs](https://img.shields.io/badge/Next.js-15-000?style=flat-square&logo=nextdotjs&logoColor=white)
![typescript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=flat-square&logo=typescript&logoColor=white)
![threejs](https://img.shields.io/badge/Three.js-r3f-000?style=flat-square&logo=threedotjs&logoColor=white)
![tailwind](https://img.shields.io/badge/Tailwind-4-38BDF8?style=flat-square&logo=tailwindcss&logoColor=white)
![fastapi](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)
![umap](https://img.shields.io/badge/UMAP--learn-dim%20reduction-4A154B?style=flat-square)
![hdbscan](https://img.shields.io/badge/HDBSCAN-clustering-222?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-A31F34?style=flat-square)
![ci](https://github.com/koprjaa/chromadb-embedding-visualizer/actions/workflows/ci.yml/badge.svg)

<!-- TODO: drag-drop demo video here via GitHub web editor -->

If you've ever squinted at a `t-SNE` PNG to understand whether your RAG embeddings actually cluster the way you want them to, this is the nicer version of that. Load a ChromaDB database, run UMAP down to 3D + HDBSCAN for cluster labels, serve the result as cached JSON, render it as a spinning point cloud with hover-to-inspect-doc tooltips.

Built originally to debug the embeddings in [protext-scraper](https://github.com/koprjaa/protext-scraper) → press releases embedded with `BAAI/bge-m3`.

## How it's split

```
├── src/                      # Next.js 15 frontend (App Router)
│   ├── app/
│   │   ├── page.tsx          # Three.js scene + hover/tooltip
│   │   ├── layout.tsx
│   │   └── globals.css       # Tailwind v4
│   ├── components/ui/        # shadcn-style primitives
│   └── lib/
└── backend/                  # Python FastAPI
    ├── precompute.py         # ChromaDB → UMAP(3D) → HDBSCAN → embeddings_cache.json
    ├── main.py               # serves the JSON + doc-metadata endpoint
    └── requirements.txt
```

Frontend ships the cached coordinates once, then navigates entirely client-side. No round-trip per hover, no re-computation per session. Dark/light theme, orbit controls.

## Run it

**Backend:**

```bash
cd backend
uv venv
uv pip install -r requirements.txt
# put a ChromaDB folder at ./data/chroma-db/protext/ (or set CHROMA_DB_PATH)
python precompute.py     # generates embeddings_cache.json with 3D coords + cluster labels
python main.py           # FastAPI on :8000
```

**Frontend:**

```bash
npm install
npm run dev              # http://localhost:3000
```

`CHROMA_DB_PATH` env var overrides the default data folder:

```bash
export CHROMA_DB_PATH="/absolute/path/to/your/chroma-db"
```

## What the precompute does

1. Reads every document + embedding from the ChromaDB collection
2. UMAP reduces the high-dim vectors (1024-dim for `bge-m3`) down to 3 components
3. HDBSCAN labels clusters in the reduced space
4. Writes `embeddings_cache.json` with `[x, y, z, cluster_id, doc_id, metadata]` per point

HDBSCAN here isn't the source of truth for clustering — it's a label-for-colouring so the scene reads visually.

## Why UMAP + HDBSCAN, not t-SNE

UMAP preserves both local *and* global structure better than t-SNE, so points that are truly far apart in the original space stay far apart in 3D. HDBSCAN over UMAP is a well-known pairing precisely because HDBSCAN tends to fall apart on the raw high-dim space but works well on the UMAP output.

## Known limits

- **Pre-compute is not live.** The frontend reads a snapshot; re-run `precompute.py` when the ChromaDB collection changes.
- **Memory-bound on ingest** — UMAP has to hold the full embedding matrix in RAM.
- **One collection at a time.** Multi-collection overlays would need a small refactor in `precompute.py`.

## License

[MIT](LICENSE)
