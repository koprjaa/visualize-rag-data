"""
Pre-compute 3D coordinates from embeddings using UMAP with optimized parameters.
Based on research: https://umap-learn.readthedocs.io/en/latest/parameters.html
"""

import json
import struct
import sqlite3
import pickle
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed, using PCA")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("HDBSCAN not installed, skipping clustering")

CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db_PROTEXT"
SQLITE_PATH = CHROMA_DB_PATH / "chroma.sqlite3"
CACHE_FILE = Path(__file__).parent / "embeddings_cache.json"


def get_documents(limit=500000):
    """Get documents from SQLite with embedding_id for mapping."""
    conn = sqlite3.connect(str(SQLITE_PATH))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT e.id, e.embedding_id, 
               MAX(CASE WHEN em.key = 'chroma:document' THEN em.string_value END) as document,
               MAX(CASE WHEN em.key = 'title' THEN em.string_value END) as title,
               MAX(CASE WHEN em.key = 'category' THEN em.string_value END) as category
        FROM embeddings e
        LEFT JOIN embedding_metadata em ON e.id = em.id
        GROUP BY e.id, e.embedding_id
        ORDER BY e.id
        LIMIT ?
    """, (limit,))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            "db_id": row[0],
            "id": row[1],  # This is the UUID (embedding_id)
            "text": (row[2] or "")[:300],
            "title": row[3] or "",
            "category": row[4] or ""
        })
    
    conn.close()
    return results


def load_id_mapping():
    """Load the UUID to HNSW label mapping from pickle."""
    collection_folders = [d for d in CHROMA_DB_PATH.iterdir() 
                         if d.is_dir() and (d / "index_metadata.pickle").exists()]
    
    if not collection_folders:
        raise Exception("No index_metadata.pickle found")
    
    pickle_file = collection_folders[0] / "index_metadata.pickle"
    print(f"Loading ID mapping from {pickle_file}...")
    
    with open(pickle_file, "rb") as f:
        metadata = pickle.load(f)
    
    # id_to_label: UUID -> HNSW internal label (index in binary file)
    id_to_label = metadata.get("id_to_label", {})
    print(f"   Loaded mapping for {len(id_to_label)} embeddings")
    
    return id_to_label


def read_all_embeddings(dimensions=1024):
    """Read ALL embeddings from HNSW binary file into a dict by label."""
    collection_folders = [d for d in CHROMA_DB_PATH.iterdir() 
                         if d.is_dir() and (d / "data_level0.bin").exists()]
    
    if not collection_folders:
        raise Exception("No HNSW index found")
    
    data_file = collection_folders[0] / "data_level0.bin"
    
    print(f"Reading embeddings from {data_file}...")
    
    with open(data_file, "rb") as f:
        data = f.read()
    
    M = 32  # max_neighbors (from header.bin)
    entry_size = 4236  # From header.bin - includes 8 bytes padding after vector
    
    num_vectors = len(data) // entry_size
    print(f"   Found {num_vectors} vectors in binary file")
    
    # Read all embeddings indexed by their position (label)
    embeddings = {}
    for label in range(num_vectors):
        if label % 100000 == 0:
            print(f"  Reading {label}/{num_vectors}...")
        offset = label * entry_size
        vector_offset = offset + 4 + M * 4
        if vector_offset + dimensions * 4 <= len(data):
            embeddings[label] = np.frombuffer(
                data[vector_offset:vector_offset + dimensions * 4], 
                dtype=np.float32
            ).copy()
    
    return embeddings


def main():
    print("=" * 60)
    print("Pre-computing 3D coordinates for visualization")
    print("=" * 60)
    
    # Load ID mapping first
    print("\n1. Loading UUID to HNSW label mapping...")
    id_to_label = load_id_mapping()
    
    # Get documents
    print("\n2. Loading documents from SQLite...")
    all_docs = get_documents(limit=10000)  # Test with 10k first
    print(f"   Loaded {len(all_docs)} documents")
    
    # Read ALL embeddings from binary file
    print("\n3. Reading ALL embeddings from binary file...")
    all_embeddings = read_all_embeddings()
    print(f"   Read {len(all_embeddings)} embeddings")
    
    # Match documents to embeddings using UUID mapping
    print("\n4. Matching documents to embeddings via UUID...")
    docs = []
    embeddings_list = []
    missing = 0
    
    for doc in all_docs:
        uuid = doc["id"]
        if uuid in id_to_label:
            label = id_to_label[uuid]
            if label in all_embeddings:
                docs.append(doc)
                embeddings_list.append(all_embeddings[label])
            else:
                missing += 1
        else:
            missing += 1
    
    print(f"   Matched {len(docs)} documents, {missing} missing")
    
    embeddings = np.array(embeddings_list, dtype=np.float32)
    actual_count = len(docs)
    
    # Normalize embeddings (important for UMAP with cosine metric)
    print("\n5. Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings = embeddings / norms
    
    # UMAP directly on full embeddings (no PCA - preserves cosine similarity)
    if HAS_UMAP:
        print("\n6. Computing UMAP reduction to 3D (no PCA)...")
        print("   Using full 1024-dim embeddings with cosine metric")
        print("   - n_neighbors=15 (preserve local similarity)")
        print("   - min_dist=0.1 (slight separation)")
        
        reducer = UMAP(
            n_components=3,       # 3D for exploration
            n_neighbors=15,       # Preserve local structure
            min_dist=0.1,         # Slight separation
            spread=1.0,           # Natural spread
            metric='cosine',      # Matches embedding similarity
            init='spectral',      # Preserves global structure
            random_state=42,      # Reproducibility
            low_memory=True,
            verbose=True
        )
        coords_3d = reducer.fit_transform(embeddings)  # Full embeddings, no PCA
    else:
        print("\n6. Computing PCA reduction to 3D (UMAP not available)...")
        pca3 = PCA(n_components=3, random_state=42)
        coords_3d = pca3.fit_transform(embeddings)
    
    # Scale coordinates preserving natural density (single scale factor)
    print("\n7. Scaling coordinates (preserving natural density)...")
    # Center at origin
    coords_3d = coords_3d - coords_3d.mean(axis=0)
    # Scale so max distance from center is ~3 (good for viewing)
    max_dist = np.max(np.linalg.norm(coords_3d, axis=1))
    if max_dist > 0:
        coords_3d = coords_3d * (3.0 / max_dist)
    
    # Clustering for colors
    cluster_labels = np.zeros(actual_count, dtype=int)
    if HAS_HDBSCAN:
        print("\n8. Computing HDBSCAN clusters for coloring...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,           # Smaller = more clusters
            min_samples=5,                 # Smaller = more clusters
            cluster_selection_epsilon=0.0, # No epsilon limit
            cluster_selection_method='leaf'  # More granular clusters
        )
        cluster_labels = clusterer.fit_predict(coords_3d)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"   Found {n_clusters} clusters")
    
    # Build cache
    print("\n9. Building cache...")
    cache = {
        "count": actual_count,
        "dimensions": 1024,
        "model": "BAAI/bge-m3",
        "method": "UMAP" if HAS_UMAP else "PCA",
        "points": []
    }
    
    for i in range(actual_count):
        if i % 50000 == 0:
            print(f"   Processing {i}/{actual_count}...")
        doc = docs[i]
        cache["points"].append({
            "id": doc["id"],
            "text": doc["text"],
            "title": doc["title"],
            "category": doc.get("category", ""),
            "cluster": int(cluster_labels[i]),
            "x": float(coords_3d[i, 0]),
            "y": float(coords_3d[i, 1]),
            "z": float(coords_3d[i, 2])
        })
    
    # Save to file
    print(f"\n10. Saving cache to {CACHE_FILE}...")
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"Done! Cache saved with {len(cache['points'])} points")
    print(f"File size: {CACHE_FILE.stat().st_size / 1024 / 1024:.2f} MB")
    if HAS_HDBSCAN:
        print(f"Clusters found: {n_clusters}")
    print("=" * 60)


if __name__ == "__main__":
    main()
