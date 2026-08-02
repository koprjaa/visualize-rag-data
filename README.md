# chromadb-embedding-visualizer

Renders a ChromaDB vector store as a 3D point cloud in the browser. UMAP and HDBSCAN run in a Python backend, Three.js draws the scene, and the frontend loads pre-computed coordinates.

![nextjs](https://img.shields.io/badge/Next.js-16-000?style=flat-square&logo=nextdotjs&logoColor=white)
![python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![license](https://img.shields.io/badge/license-MIT-A31F34?style=flat-square)
[![ci](https://github.com/koprjaa/chromadb-embedding-visualizer/actions/workflows/ci.yml/badge.svg)](https://github.com/koprjaa/chromadb-embedding-visualizer/actions/workflows/ci.yml)

https://github.com/user-attachments/assets/65218844-a127-4740-ab35-1d95218e9b30

A static t-SNE image tells you little about whether your RAG embeddings cluster the way you expect. This tool loads a ChromaDB database, reduces it to three dimensions with UMAP, labels the clusters with HDBSCAN, caches the result as JSON, and renders it as a point cloud with a tooltip on hover.

It was built to inspect the embeddings in [protext-scraper](https://github.com/koprjaa/protext-scraper), where press releases are embedded with `BAAI/bge-m3`.

## Install

Backend:

```bash
cd backend
uv venv
uv pip install -r requirements.txt
```

Frontend:

```bash
npm install
```

## Use

Place a ChromaDB folder at `./data/chroma-db/protext/`, or set `CHROMA_DB_PATH` to another location:

```bash
export CHROMA_DB_PATH="/absolute/path/to/your/chroma-db"
```

Generate the cache, then start both processes:

```bash
cd backend
python precompute.py     # writes embeddings_cache.json
python main.py           # FastAPI on port 8000
```

```bash
npm run dev              # http://localhost:3000
```

## How it works

```
src/                    Next.js 16 frontend, App Router
  app/page.tsx          Three.js scene, hover and tooltip
  app/globals.css       Tailwind v4
  components/ui/        shadcn style primitives
backend/
  precompute.py         ChromaDB -> UMAP 3D -> HDBSCAN -> embeddings_cache.json
  main.py               Serves the JSON and the document metadata
```

`precompute.py` runs four steps.

1. Read every document and embedding from the ChromaDB collection.
2. Reduce the vectors to three components with UMAP. The `bge-m3` model produces 1024 dimensions.
3. Label the clusters with HDBSCAN in the reduced space.
4. Write `embeddings_cache.json` with `[x, y, z, cluster_id, doc_id, metadata]` per point.

HDBSCAN here assigns colors. It is not the authority on how the documents cluster.

UMAP keeps both local and global structure better than t-SNE, so points that are far apart in the original space stay far apart in 3D. HDBSCAN pairs well with UMAP, because it degrades on raw high dimensional data and works on the reduced output.

The frontend loads the cached coordinates once and then runs client side. There is no request per hover and no recomputation per session. The scene supports orbit controls and a dark and light theme.

## Development

```bash
uv run --extra dev ruff check .
uv run --extra dev pytest -q
```

The backend suite builds an HNSW index byte by byte, so it needs no ChromaDB
database. It covers the header offsets, the label to position mapping, and the
coordinate scaling. CI runs ruff and pytest on Python 3.11 and 3.12, on Linux
and Windows, next to the frontend lint and typecheck.

## Limits

- The view is a snapshot, not a live query. Run `precompute.py` again after the collection changes.
- UMAP holds the full embedding matrix in memory during ingest.
- One collection at a time. An overlay of several collections needs a change in `precompute.py`.
- The HNSW header layout is undocumented. The byte offsets in `backend/chroma_io.py` were read off a real index and a ChromaDB upgrade could move them.

## License

[MIT](LICENSE)
