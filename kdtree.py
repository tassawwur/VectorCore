import numpy as np
from typing import List, Tuple, Optional, Any
import heapq


class KDNode:
    """A node in the k-d tree."""
    
    def __init__(self, vector: np.ndarray, doc_id: str, axis: int = 0):
        self.vector = vector
        self.doc_id = doc_id
        self.axis = axis
        self.left: Optional['KDNode'] = None
        self.right: Optional['KDNode'] = None


class KDTree:
    """
    K-D Tree implementation for fast nearest neighbor search in vector space.
    
    This data structure allows us to search through millions of vectors much faster
    than brute force by organizing the space into a binary tree where each level
    splits on a different dimension.
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.root: Optional[KDNode] = None
        self.size = 0
    
    def insert(self, vector: np.ndarray, doc_id: str):
        """Insert a vector with its document ID into the k-d tree."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match tree dimension {self.dimension}")
        
        vector = np.array(vector, dtype=np.float32)
        self.root = self._insert_recursive(self.root, vector, doc_id, 0)
        self.size += 1
    
    def _insert_recursive(self, node: Optional[KDNode], vector: np.ndarray, doc_id: str, depth: int) -> KDNode:
        """Recursively insert a node into the k-d tree."""
        if node is None:
            return KDNode(vector, doc_id, depth % self.dimension)
        
        axis = depth % self.dimension
        
        if vector[axis] < node.vector[axis]:
            node.left = self._insert_recursive(node.left, vector, doc_id, depth + 1)
        else:
            node.right = self._insert_recursive(node.right, vector, doc_id, depth + 1)
        
        return node
    
    def search_nearest(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Find the k nearest neighbors to the query vector.
        
        Returns a list of (doc_id, distance) tuples, sorted by distance.
        """
        if self.root is None:
            return []
        
        query_vector = np.array(query_vector, dtype=np.float32)
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(query_vector)} doesn't match tree dimension {self.dimension}")
        
        # Use a max heap to keep track of the k closest points
        # We negate distances because heapq is a min heap
        best_matches = []
        
        def search_recursive(node: Optional[KDNode], depth: int = 0):
            if node is None:
                return
            
            # Calculate distance to current node
            distance = self._euclidean_distance(query_vector, node.vector)
            
            # Add to best matches if we have room or if it's better than our worst match
            if len(best_matches) < k:
                heapq.heappush(best_matches, (-distance, node.doc_id, distance))
            elif distance < best_matches[0][2]:  # Compare with worst distance
                heapq.heapreplace(best_matches, (-distance, node.doc_id, distance))
            
            # Determine which side to search first
            axis = depth % self.dimension
            diff = query_vector[axis] - node.vector[axis]
            
            if diff < 0:
                near_node, far_node = node.left, node.right
            else:
                near_node, far_node = node.right, node.left
            
            # Search the near side
            search_recursive(near_node, depth + 1)
            
            # Check if we need to search the far side
            # Only search if the distance to the splitting plane is less than our current worst distance
            if len(best_matches) < k or abs(diff) < best_matches[0][2]:
                search_recursive(far_node, depth + 1)
        
        search_recursive(self.root)
        
        # Sort by distance and return doc_ids with distances
        result = [(doc_id, distance) for _, doc_id, distance in sorted(best_matches, key=lambda x: x[2])]
        return result
    
    def _euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors."""
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))
    
    def get_all_vectors(self) -> List[Tuple[str, np.ndarray]]:
        """Get all vectors stored in the tree for persistence."""
        result = []
        
        def traverse(node: Optional[KDNode]):
            if node is not None:
                result.append((node.doc_id, node.vector))
                traverse(node.left)
                traverse(node.right)
        
        traverse(self.root)
        return result 