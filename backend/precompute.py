"""
Pre-compute 3D coordinates from embeddings using UMAP with optimized parameters.
Based on research: https://umap-learn.readthedocs.io/en/latest/parameters.html
"""

import json
import struct
import sqlite3
import pickle
import sys
import os
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def log(msg):
    """Print with immediate flush."""
    print(msg, flush=True)

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    log("UMAP not installed, using PCA")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    log("HDBSCAN not installed, skipping clustering")

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sklearn.feature_extraction.text import CountVectorizer
    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False
    log("BERTopic not installed, skipping topic modeling")

def resolve_chroma_db_path() -> Path:
    """
    Resolve ChromaDB path.
    Priority:
    1) CHROMA_DB_PATH env var
    2) new default: <repo>/data/chroma-db/protext
    3) legacy default: <repo>/chroma_db_PROTEXT
    """
    env = os.getenv("CHROMA_DB_PATH")
    if env:
        return Path(env).expanduser()

    repo_root = Path(__file__).resolve().parent.parent
    new_default = repo_root / "data" / "chroma-db" / "protext"
    legacy_default = repo_root / "chroma_db_PROTEXT"

    if new_default.exists():
        return new_default
    if legacy_default.exists():
        return legacy_default
    return new_default


CHROMA_DB_PATH = resolve_chroma_db_path()
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
    header_file = collection_folders[0] / "header.bin"
    
    print(f"Reading embeddings from {data_file}...")
    
    # Read header to get parameters
    with open(header_file, "rb") as f:
        header = f.read()
    
    import struct
    num_vectors = struct.unpack('<I', header[20:24])[0]
    # NOTE: entry_size is at offset 28, NOT 36!
    entry_size = struct.unpack('<I', header[28:32])[0]
    vector_offset_in_entry = struct.unpack('<I', header[44:48])[0]
    
    print(f"   Header: {num_vectors} vectors, entry_size={entry_size}, vector_offset={vector_offset_in_entry}")
    
    with open(data_file, "rb") as f:
        data = f.read()
    
    # Data starts at offset 0 (no file header)
    file_size = len(data)
    
    # Read all embeddings indexed by their position (label)
    # Labels start at 1, but position in file starts at 0
    # So label N is at position N-1
    embeddings = {}
    max_pos = file_size // entry_size
    print(f"   File contains {max_pos} vector slots")
    
    for label in tqdm(range(1, num_vectors + 1), desc="   Reading embeddings", unit="vec"):
        position = label - 1  # Labels start at 1, positions at 0
        offset = position * entry_size + vector_offset_in_entry
        if offset + dimensions * 4 <= file_size:
            embeddings[label] = np.frombuffer(
                data[offset:offset + dimensions * 4], 
                dtype=np.float32
            ).copy()
    
    return embeddings


def main():
    log("=" * 60)
    log("Pre-computing 3D coordinates for visualization")
    log("=" * 60)
    
    # Load ID mapping first
    log("\n1. Loading UUID to HNSW label mapping...")
    id_to_label = load_id_mapping()
    
    # Get documents
    log("\n2. Loading documents from SQLite...")
    all_docs = get_documents(limit=500000)  # Full dataset
    print(f"   Loaded {len(all_docs)} documents")
    
    # Read ALL embeddings from binary file
    log("\n3. Reading ALL embeddings from binary file...")
    all_embeddings = read_all_embeddings()
    print(f"   Read {len(all_embeddings)} embeddings")
    
    # Match documents to embeddings using UUID mapping
    log("\n4. Matching documents to embeddings via UUID...")
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
    
    # We'll use TF-IDF for BOTH visualization AND clustering
    # This ensures documents with same topic are close in 3D space
    cluster_labels = np.zeros(actual_count, dtype=int)
    topic_info = {}  # topic_id -> {name, keywords}
    coords_3d = None
    
    if HAS_BERTOPIC:
        log("\n5. Running TF-IDF based visualization and clustering...")
        log("   Using SPARSE vectors (keyword-based) for BOTH:")
        log("   - 3D positions (topics will be spatially grouped)")
        log("   - Cluster assignments (by shared keywords)")
        
        # Get document texts
        doc_texts = [doc["text"] for doc in docs]
        
        # TF-IDF Vectorizer for sparse representation
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        tfidf_vectorizer = TfidfVectorizer(
            strip_accents=None,     # Preserve Czech diacritics
            token_pattern=r'(?u)\b[^\W\d_]{2,}\b',  # Unicode letters only
            lowercase=True,
            min_df=5,
            max_df=0.5,
            ngram_range=(1, 2),
            max_features=5000  # Limit vocabulary for speed
        )
        
        log("   Creating TF-IDF matrix...")
        tfidf_matrix = tfidf_vectorizer.fit_transform(doc_texts)
        print(f"   TF-IDF matrix: {tfidf_matrix.shape}")
        
        # Convert sparse to dense for UMAP (required)
        tfidf_dense = tfidf_matrix.toarray()
        
        # UMAP to 3D for visualization (on TF-IDF, not dense embeddings!)
        log("\n6. Reducing TF-IDF to 3D for visualization...")
        sys.stdout.flush()
        umap_3d = UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            spread=1.0,
            n_jobs=-1,  # Use all CPU cores
            low_memory=True,
            metric='cosine',
            random_state=42,
            verbose=True
        )
        coords_3d = umap_3d.fit_transform(tfidf_dense)
        
        # Scale coordinates
        log("\n7. Scaling coordinates...")
        coords_3d = coords_3d - coords_3d.mean(axis=0)
        max_dist = np.max(np.linalg.norm(coords_3d, axis=1))
        if max_dist > 0:
            coords_3d = coords_3d * (3.0 / max_dist)
        
        # UMAP to 5D for clustering
        log("\n8. Reducing TF-IDF to 5D for clustering...")
        sys.stdout.flush()
        umap_5d = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            n_jobs=-1,
            low_memory=True,
            random_state=42,
            verbose=False
        )
        tfidf_reduced = umap_5d.fit_transform(tfidf_dense)
        
        # Czech stopwords - common generic words to exclude from topic names
        czech_stopwords = [
            # Pronouns & articles
            'jsem', 'jsi', 'jsou', 'byl', 'byla', 'bylo', 'byli', 'být', 'bude', 'budou',
            'mám', 'máme', 'mají', 'může', 'mohou', 'musí', 'chce', 'chtějí',
            # Common verbs
            'mít', 'dát', 'jít', 'říci', 'vidět', 'stát', 'dostat', 'chtít', 'vědět',
            'dělat', 'udělat', 'použít', 'ukázat', 'ukazuje', 'možné', 'ideální',
            # Prepositions & conjunctions  
            'přes', 'podle', 'nebo', 'ale', 'tak', 'jen', 'již', 'ještě', 'také',
            'proto', 'tedy', 'když', 'kde', 'jak', 'proč', 'který', 'která', 'které',
            # Numbers & quantities
            'jeden', 'dva', 'tři', 'první', 'druhý', 'třetí', 'celý', 'celém', 'polovina',
            'více', 'méně', 'hodně', 'málo', 'korun', 'procent', 'tisíc', 'milion',
            # Generic nouns
            'rok', 'den', 'čas', 'místo', 'způsob', 'část', 'strana', 'věc', 'člověk',
            'lidé', 'práce', 'svět', 'život', 'firma', 'společnost', 'funkce',
            # Web/tech generic
            'www', 'http', 'https', 'com', 'org', 'cz', 'html',
            'the', 'and', 'for', 'you', 'are', 'this', 'that', 'our', 'your', 'their',
            'with', 'have', 'from', 'will', 'can', 'all', 'new', 'more', 'not', 'was',
            # Other common Czech
            'právě', 'zatím', 'nyní', 'dnes', 'včera', 'zítra', 'letos', 'příští',
            'vlastně', 'nejen', 'totiž', 'ovšem', 'pouze', 'hlavně', 'třeba', 'bohužel',
            'samozřejmě', 'skutečně', 'dokonce', 'nakonec', 'opravdu', 'rozhodně',
            'prostě', 'zkrátka', 'nicméně', 'přitom', 'dokud', 'jakmile', 'pokud',
            'pro', 'nad', 'pod', 'před', 'mezi', 'vedle', 'kolem', 'během', 'pomocí',
            'světě', 'světa', 'věku', 'době', 'rámci', 'případě', 'oblasti', 'základě',
            # More generic words found in topics
            'informací', 'informace', 'company', 'rozdíl', 'aby', 'není', 'jsme', 'než',
            'cílem', 'cíle', 'výši', 'styl', 'mladí', 'mladý', 'republika',
            'nový', 'nová', 'nové', 'velký', 'velká', 'malý', 'malá', 'dobrý', 'dobrá',
            'každý', 'každá', 'další', 'jiný', 'jiná', 'stejný', 'stejná', 'různý', 'různá',
            # Even more generic
            'kontakt', 'spolu', 'roce', 'roku', 'let',
            'bez', 'počet', 'chtěli', 'chtěl', 'naším', 'naše', 'náš', 'meziroční',
            'dětí', 'petra', 'petr', 'chtěla', 'mohli', 'měli', 'říká', 'uvádí',
        ]
        
        # Use BERTopic with our TF-IDF vectors
        vectorizer = CountVectorizer(
            strip_accents=None,
            token_pattern=r'(?u)\b[^\W\d_]{3,}\b',  # Min 3 chars
            lowercase=True,
            min_df=2,
            max_df=1.0,
            ngram_range=(1, 1),
            stop_words=czech_stopwords  # Filter generic words
        )
        
        # HDBSCAN - moderate settings for ~20-30 topics
        hdbscan_for_bertopic = hdbscan.HDBSCAN(
            min_cluster_size=80,    # ~80 docs per cluster minimum
            min_samples=5,
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # UMAP for BERTopic
        umap_for_bertopic = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            n_jobs=-1,
            low_memory=True,
            random_state=42
        )
        
        topic_model = BERTopic(
            language="multilingual",
            embedding_model=None,
            umap_model=umap_for_bertopic,
            hdbscan_model=hdbscan_for_bertopic,
            vectorizer_model=vectorizer,
            top_n_words=5,
            calculate_probabilities=False,
            verbose=True
        )
        
        # Fit with TF-IDF dense vectors
        topics, _ = topic_model.fit_transform(doc_texts, embeddings=tfidf_dense)
        cluster_labels = np.array(topics)
        
        # Get topic info
        topic_df = topic_model.get_topic_info()
        n_clusters = len(topic_df) - 1  # -1 for outlier topic
        print(f"   Found {n_clusters} topics")
        
        # Build topic info dict
        for _, row in topic_df.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                topic_info[-1] = {"name": "Nezařazeno", "keywords": []}
            else:
                # Get top keywords for this topic
                topic_words = topic_model.get_topic(topic_id)
                if topic_words:
                    keywords = [word for word, _ in topic_words[:5]]
                    # Use just the top keyword as the topic name
                    name = keywords[0].title() if keywords else f"Topic {topic_id}"
                    topic_info[topic_id] = {"name": name, "keywords": keywords}
        
        print(f"\n   Topic summary:")
        for tid, info in sorted(topic_info.items())[:10]:
            print(f"   Topic {tid}: {info['name']}")
        if len(topic_info) > 10:
            print(f"   ... and {len(topic_info) - 10} more topics")
            
    elif HAS_HDBSCAN:
        log("\n8. Computing HDBSCAN clusters for coloring (no BERTopic)...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=100,
            min_samples=5,
            cluster_selection_epsilon=0.1,
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(coords_3d)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"   Found {n_clusters} clusters")
    
    # Build cache
    log("\n9. Building cache...")
    cache = {
        "count": actual_count,
        "dimensions": 1024,
        "model": "BAAI/bge-m3",
        "method": "TF-IDF + UMAP + BERTopic" if HAS_BERTOPIC else ("UMAP" if HAS_UMAP else "PCA"),
        "topics": topic_info,  # Topic ID -> {name, keywords}
        "points": []
    }
    
    for i in tqdm(range(actual_count), desc="   Building cache", unit="doc"):
        doc = docs[i]
        topic_id = int(cluster_labels[i])
        topic_name = topic_info.get(topic_id, {}).get("name", f"Topic {topic_id}")
        cache["points"].append({
            "id": doc["id"],
            "text": doc["text"],
            "title": doc["title"],
            "category": doc.get("category", ""),
            "cluster": topic_id,
            "topic_name": topic_name,
            "x": float(coords_3d[i, 0]),
            "y": float(coords_3d[i, 1]),
            "z": float(coords_3d[i, 2])
        })
    
    # Save to file
    print(f"\n10. Saving cache to {CACHE_FILE}...")
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    
    log("\n" + "=" * 60)
    print(f"Done! Cache saved with {len(cache['points'])} points")
    print(f"File size: {CACHE_FILE.stat().st_size / 1024 / 1024:.2f} MB")
    if HAS_HDBSCAN:
        print(f"Clusters found: {n_clusters}")
    log("=" * 60)


if __name__ == "__main__":
    main()
