"""
FastAPI backend for ChromaDB visualization.
Serves pre-computed 3D coordinates from cache file for instant loading.
"""

import json
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ChromaDB Visualizer API")

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache file path
CACHE_FILE = Path(__file__).parent / "embeddings_cache.json"
CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db_PROTEXT"

# Load cache on startup
_cache = None


def load_cache():
    global _cache
    if _cache is None:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        else:
            _cache = {"count": 0, "points": [], "model": "unknown", "dimensions": 0}
    return _cache


class EmbeddingPoint(BaseModel):
    id: str
    text: str
    x: float
    y: float
    z: float
    cluster: int = 0
    topic_name: str = ""
    metadata: dict | None = None


class TopicInfo(BaseModel):
    id: int
    name: str
    keywords: list[str]
    count: int = 0


class CollectionInfo(BaseModel):
    name: str
    count: int
    dimensions: int
    model: str
    topics: list[TopicInfo] = []


@app.get("/api/info")
async def get_info() -> CollectionInfo:
    """Get information about the ChromaDB collection."""
    cache = load_cache()
    
    # Also read from meta file for total count
    meta_path = CHROMA_DB_PATH / "embedding_meta.json"
    model = cache.get("model", "unknown")
    dimensions = cache.get("dimensions", 1024)
    
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            model = meta.get("model_name", model)
            dimensions = meta.get("dimensions", dimensions)
    
    # Build topics list with counts
    topics_raw = cache.get("topics", {})
    points = cache.get("points", [])
    
    # Count documents per topic
    topic_counts = {}
    for p in points:
        tid = p.get("cluster", 0)
        topic_counts[tid] = topic_counts.get(tid, 0) + 1
    
    topics = []
    for tid_str, info in topics_raw.items():
        tid = int(tid_str)
        topics.append(TopicInfo(
            id=tid,
            name=info.get("name", f"Topic {tid}"),
            keywords=info.get("keywords", []),
            count=topic_counts.get(tid, 0)
        ))
    
    # Sort by count descending, but put -1 (outliers) last
    topics.sort(key=lambda t: (t.id == -1, -t.count))
    
    return CollectionInfo(
        name="langchain",
        count=len(points),
        dimensions=dimensions,
        model=model,
        topics=topics
    )


@app.get("/api/embeddings")
async def get_embeddings(
    limit: int = Query(default=1000, le=500000),
    offset: int = Query(default=0, ge=0)
) -> list[EmbeddingPoint]:
    """
    Get pre-computed embeddings with 3D coordinates.
    Instant loading from cache file.
    """
    cache = load_cache()
    points = cache.get("points", [])
    
    # Slice based on limit and offset
    selected = points[offset:offset + limit]
    
    return [
        EmbeddingPoint(
            id=p["id"],
            text=p.get("text", ""),
            x=p["x"],
            y=p["y"],
            z=p["z"],
            cluster=p.get("cluster", 0),
            topic_name=p.get("topic_name", ""),
            metadata={
                "title": p.get("title", ""),
                "category": p.get("category", "")
            }
        )
        for p in selected
    ]


if __name__ == "__main__":
    import uvicorn
    # Pre-load cache
    load_cache()
    print(f"Cache loaded with {len(_cache.get('points', []))} points")
    uvicorn.run(app, host="0.0.0.0", port=8000)
