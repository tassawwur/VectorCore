# import json  # Reserved for future use
import pickle

# import socket  # Reserved for future use
import threading
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from kdtree import KDTree


class VectorCore:
    """
    High-performance in-memory vector database with k-d tree indexing.

    Supports operations:
    - ADD: Store vectors with document IDs
    - QUERY: Find k nearest neighbors to a query vector
    - SAVE: Persist database to disk
    - LOAD: Load database from disk
    - STATS: Get database statistics
    """

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension
        self.kdtree: Optional[KDTree] = None
        self.vectors: Dict[str, np.ndarray] = {}  # Backup storage for exact lookups
        self.lock = threading.RLock()  # Thread-safe operations
        self.start_time = time.time()

    def add_vector(self, doc_id: str, vector: List[float]) -> bool:
        """Add a vector to the database."""
        with self.lock:
            try:
                vector_array = np.array(vector, dtype=np.float32)

                # Initialize dimension on first vector
                if self.dimension is None:
                    self.dimension = len(vector_array)
                    self.kdtree = KDTree(self.dimension)
                    print(f"Initialized VectorCore with dimension: {self.dimension}")

                # Validate dimension
                if len(vector_array) != self.dimension:
                    raise ValueError(
                        f"Vector dimension {len(vector_array)} doesn't match database dimension {self.dimension}"
                    )

                # Update existing vector or add new one
                if doc_id in self.vectors:
                    print(f"Warning: Updating existing vector for doc_id: {doc_id}")
                    # For updates, we'd need to rebuild the tree or implement tree updates
                    # For simplicity, we'll rebuild the tree (not optimal for production)
                    self.vectors[doc_id] = vector_array
                    self._rebuild_tree()
                else:
                    self.vectors[doc_id] = vector_array
                    if self.kdtree is not None:
                        self.kdtree.insert(vector_array, doc_id)

                return True

            except Exception as e:
                print(f"Error adding vector for {doc_id}: {e}")
                return False

    def query_similar(self, query_vector: List[float], k: int = 10) -> List[str]:
        """Find k most similar vectors to the query vector."""
        with self.lock:
            try:
                if self.kdtree is None or self.kdtree.size == 0:
                    return []

                query_array = np.array(query_vector, dtype=np.float32)

                if len(query_array) != self.dimension:
                    raise ValueError(
                        f"Query vector dimension {len(query_array)} doesn't match database dimension {self.dimension}"
                    )

                # Limit k to available vectors
                k = min(k, self.kdtree.size)

                # Get nearest neighbors with distances
                results = self.kdtree.search_nearest(query_array, k)

                # Return just the doc_ids
                return [doc_id for doc_id, distance in results]

            except Exception as e:
                print(f"Error querying similar vectors: {e}")
                return []

    def query_similar_with_distances(
        self, query_vector: List[float], k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find k most similar vectors with their distances."""
        with self.lock:
            try:
                if self.kdtree is None or self.kdtree.size == 0:
                    return []

                query_array = np.array(query_vector, dtype=np.float32)

                if len(query_array) != self.dimension:
                    raise ValueError(
                        f"Query vector dimension {len(query_array)} doesn't match database dimension {self.dimension}"
                    )

                k = min(k, self.kdtree.size)
                return self.kdtree.search_nearest(query_array, k)

            except Exception as e:
                print(f"Error querying similar vectors: {e}")
                return []

    def _rebuild_tree(self):
        """Rebuild the k-d tree from scratch (used for updates)."""
        if self.dimension is None:
            return

        self.kdtree = KDTree(self.dimension)
        for doc_id, vector in self.vectors.items():
            self.kdtree.insert(vector, doc_id)

    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self.lock:
            uptime = time.time() - self.start_time
            return {
                "vector_count": len(self.vectors),
                "dimension": self.dimension,
                "uptime_seconds": round(uptime, 2),
                "memory_usage_vectors": len(self.vectors),
                "kdtree_size": self.kdtree.size if self.kdtree else 0,
            }

    def save_to_disk(self, filename: str = "vectorcore_data.pkl") -> bool:
        """Save the vector database to disk."""
        with self.lock:
            try:
                data = {
                    "dimension": self.dimension,
                    "vectors": self.vectors,
                    "start_time": self.start_time,
                }

                with open(filename, "wb") as f:
                    pickle.dump(data, f)

                print(f"Database saved to {filename}")
                return True

            except Exception as e:
                print(f"Error saving database: {e}")
                return False

    def load_from_disk(self, filename: str = "vectorcore_data.pkl") -> bool:
        """Load the vector database from disk."""
        with self.lock:
            try:
                with open(filename, "rb") as f:
                    data = pickle.load(f)

                self.dimension = data["dimension"]
                self.vectors = data["vectors"]
                self.start_time = data.get("start_time", time.time())

                # Rebuild the k-d tree
                if self.dimension and self.vectors:
                    self.kdtree = KDTree(self.dimension)
                    for doc_id, vector in self.vectors.items():
                        self.kdtree.insert(vector, doc_id)

                print(f"Database loaded from {filename} - {len(self.vectors)} vectors")
                return True

            except FileNotFoundError:
                print(f"File {filename} not found")
                return False
            except Exception as e:
                print(f"Error loading database: {e}")
                return False

    def get_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """Get a specific vector by doc_id."""
        with self.lock:
            return self.vectors.get(doc_id)

    def remove_vector(self, doc_id: str) -> bool:
        """Remove a vector from the database."""
        with self.lock:
            if doc_id in self.vectors:
                del self.vectors[doc_id]
                # Rebuild tree after removal (not optimal, but simple)
                self._rebuild_tree()
                return True
            return False

    def clear_all(self) -> bool:
        """Clear all vectors from the database."""
        with self.lock:
            self.vectors.clear()
            self.kdtree = KDTree(self.dimension) if self.dimension else None
            return True
