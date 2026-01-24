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
├── app/                    # Next.js frontend
│   ├── page.tsx           # Main visualization component
│   ├── layout.tsx         # App layout
│   └── globals.css        # Global styles
├── backend/               # Python API
│   ├── main.py           # FastAPI server
│   ├── precompute.py     # UMAP/HDBSCAN processing
│   └── requirements.txt  # Python dependencies
├── components/ui/         # UI components (Button)
└── lib/                   # Utilities
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

Place your ChromaDB database in `chroma_db_PROTEXT/` and run:

```bash
cd backend
python precompute.py
```

This generates `embeddings_cache.json` with 3D coordinates.

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
