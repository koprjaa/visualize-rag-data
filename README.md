# ChromaDB Embedding Visualizer

Interactive 3D visualization of embeddings from a ChromaDB database using UMAP dimensionality reduction.

## Features

- 3D visualization of high-dimensional embeddings
- UMAP + HDBSCAN for dimensionality reduction and clustering
- Hover tooltips with document metadata
- Dark/Light mode toggle
- Pre-computed coordinates for instant loading

## Architecture

```
├── src/                   # Frontend source
│   ├── app/               # Next.js App Router
│   │   ├── page.tsx       # Main visualization component
│   │   ├── layout.tsx     # App layout
│   │   └── globals.css    # Global styles
│   ├── components/        # React components
│   │   └── ui/            # UI components (Button)
│   └── lib/               # Utilities
├── backend/               # Python API
│   ├── main.py           # FastAPI server
│   ├── precompute.py     # UMAP/HDBSCAN processing
│   └── requirements.txt  # Python dependencies
```

## Setup

### Frontend

```bash
npm install
npm run dev
```

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Pre-compute embeddings

Place your ChromaDB database in `data/chroma-db/protext/` (recommended) and run:

```bash
cd backend
python precompute.py
```

This generates `embeddings_cache.json` with 3D coordinates.

You can also override the database location with:

```bash
export CHROMA_DB_PATH="/absolute/path/to/your/chroma-db"
```

### Start API server

```bash
cd backend
python main.py
```

API runs on http://localhost:8000

## Tech Stack

- **Frontend**: Next.js, React Three Fiber, Three.js, Tailwind CSS
- **Backend**: FastAPI, UMAP-learn, HDBSCAN, NumPy
- **Database**: ChromaDB (BAAI/bge-m3 embeddings)
