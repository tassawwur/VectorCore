#!/usr/bin/env python3
"""
Comprehensive test suite for VectorCore database system.

This module contains unit tests, integration tests, and performance benchmarks
for all VectorCore components to ensure reliability and correctness.
"""

import unittest
import threading
import time
import tempfile
import os
import numpy as np

# from unittest.mock import patch, MagicMock  # Reserved for future use

from kdtree import KDTree
from vectorcore import VectorCore

# from server import VectorCoreServer  # Reserved for future use


class TestKDTree(unittest.TestCase):
    """Test suite for K-D Tree implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.kdtree = KDTree(dimension=3)

    def test_insert_and_search(self):
        """Test basic insert and search functionality."""
        # Insert test vectors
        self.kdtree.insert([1.0, 2.0, 3.0], "doc1")
        self.kdtree.insert([2.0, 3.0, 4.0], "doc2")
        self.kdtree.insert([3.0, 4.0, 5.0], "doc3")

        # Test search
        results = self.kdtree.search_nearest([1.1, 2.1, 3.1], k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "doc1")  # Closest should be doc1

    def test_dimension_validation(self):
        """Test dimension validation during insert."""
        with self.assertRaises(ValueError):
            self.kdtree.insert([1.0, 2.0], "doc1")  # Wrong dimension

    def test_empty_tree_search(self):
        """Test search on empty tree."""
        results = self.kdtree.search_nearest([1.0, 2.0, 3.0], k=5)
        self.assertEqual(len(results), 0)

    def test_single_node_tree(self):
        """Test tree with single node."""
        self.kdtree.insert([1.0, 2.0, 3.0], "doc1")
        results = self.kdtree.search_nearest([2.0, 3.0, 4.0], k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "doc1")

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        v1 = np.array([0, 0, 0])
        v2 = np.array([3, 4, 0])
        distance = self.kdtree._euclidean_distance(v1, v2)
        self.assertAlmostEqual(distance, 5.0, places=6)


class TestVectorCore(unittest.TestCase):
    """Test suite for VectorCore database."""

    def setUp(self):
        """Set up test fixtures."""
        self.vectorcore = VectorCore()

    def test_add_vector(self):
        """Test adding vectors to database."""
        success = self.vectorcore.add_vector("doc1", [1.0, 2.0, 3.0])
        self.assertTrue(success)
        self.assertEqual(self.vectorcore.dimension, 3)

    def test_dimension_consistency(self):
        """Test dimension consistency enforcement."""
        self.vectorcore.add_vector("doc1", [1.0, 2.0, 3.0])
        success = self.vectorcore.add_vector("doc2", [1.0, 2.0])  # Wrong dimension
        self.assertFalse(success)

    def test_query_similar(self):
        """Test similarity query functionality."""
        # Add test vectors
        self.vectorcore.add_vector("doc1", [1.0, 2.0, 3.0])
        self.vectorcore.add_vector("doc2", [2.0, 3.0, 4.0])
        self.vectorcore.add_vector("doc3", [5.0, 6.0, 7.0])

        # Query for similar vectors
        results = self.vectorcore.query_similar([1.1, 2.1, 3.1], k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("doc1", results)

    def test_get_stats(self):
        """Test statistics retrieval."""
        stats = self.vectorcore.get_stats()
        self.assertIn("vector_count", stats)
        self.assertIn("dimension", stats)
        self.assertIn("uptime_seconds", stats)

    def test_persistence(self):
        """Test save and load functionality."""
        # Add test data
        self.vectorcore.add_vector("doc1", [1.0, 2.0, 3.0])
        self.vectorcore.add_vector("doc2", [2.0, 3.0, 4.0])

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            success = self.vectorcore.save_to_disk(tmp_path)
            self.assertTrue(success)

            # Create new instance and load
            new_vectorcore = VectorCore()
            success = new_vectorcore.load_from_disk(tmp_path)
            self.assertTrue(success)

            # Verify data integrity
            self.assertEqual(new_vectorcore.dimension, 3)
            self.assertEqual(len(new_vectorcore.vectors), 2)
        finally:
            os.unlink(tmp_path)

    def test_thread_safety(self):
        """Test thread safety of concurrent operations."""

        def add_vectors(start_idx):
            for i in range(10):
                self.vectorcore.add_vector(
                    f"doc_{start_idx}_{i}", [float(i), float(i + 1), float(i + 2)]
                )

        # Initialize with first vector to set dimension
        self.vectorcore.add_vector("init", [0.0, 0.0, 0.0])

        # Run concurrent additions
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_vectors, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all vectors were added
        stats = self.vectorcore.get_stats()
        self.assertEqual(stats["vector_count"], 51)  # 1 init + 50 from threads

    def test_remove_vector(self):
        """Test vector removal functionality."""
        self.vectorcore.add_vector("doc1", [1.0, 2.0, 3.0])
        self.vectorcore.add_vector("doc2", [2.0, 3.0, 4.0])

        success = self.vectorcore.remove_vector("doc1")
        self.assertTrue(success)

        vector = self.vectorcore.get_vector("doc1")
        self.assertIsNone(vector)

        stats = self.vectorcore.get_stats()
        self.assertEqual(stats["vector_count"], 1)

    def test_clear_all(self):
        """Test clearing all vectors."""
        self.vectorcore.add_vector("doc1", [1.0, 2.0, 3.0])
        self.vectorcore.add_vector("doc2", [2.0, 3.0, 4.0])

        success = self.vectorcore.clear_all()
        self.assertTrue(success)

        stats = self.vectorcore.get_stats()
        self.assertEqual(stats["vector_count"], 0)


class TestPerformance(unittest.TestCase):
    """Performance benchmarks for VectorCore."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.vectorcore = VectorCore()

    def test_insertion_performance(self):
        """Benchmark vector insertion performance."""
        num_vectors = 1000
        dimension = 128

        start_time = time.time()
        for i in range(num_vectors):
            vector = [float(j) for j in range(dimension)]
            self.vectorcore.add_vector(f"doc_{i}", vector)

        insertion_time = time.time() - start_time
        avg_insertion_time = insertion_time / num_vectors

        print("\n📊 Insertion Performance:")
        print(f"   • {num_vectors} vectors of dimension {dimension}")
        print(f"   • Total time: {insertion_time:.3f}s")
        print(f"   • Average per vector: {avg_insertion_time:.6f}s")
        print(f"   • Throughput: {num_vectors/insertion_time:.0f} vectors/second")

        # Reasonable performance assertion
        self.assertLess(avg_insertion_time, 0.01)  # Less than 10ms per vector

    def test_query_performance(self):
        """Benchmark query performance."""
        # Insert test data
        num_vectors = 1000
        dimension = 128

        for i in range(num_vectors):
            vector = [float(j + i) for j in range(dimension)]
            self.vectorcore.add_vector(f"doc_{i}", vector)

        # Benchmark queries
        query_vector = [float(j) for j in range(dimension)]
        num_queries = 100

        start_time = time.time()
        for _ in range(num_queries):
            self.vectorcore.query_similar(
                query_vector, k=10
            )  # Results not used for timing

        query_time = time.time() - start_time
        avg_query_time = query_time / num_queries

        print("\n🔍 Query Performance:")
        print(f"   • Database: {num_vectors} vectors of dimension {dimension}")
        print(f"   • {num_queries} queries for top-10 results")
        print(f"   • Total time: {query_time:.3f}s")
        print(f"   • Average per query: {avg_query_time:.6f}s")
        print(f"   • Throughput: {num_queries/query_time:.0f} queries/second")

        # Performance assertion
        self.assertLess(avg_query_time, 0.1)  # Less than 100ms per query


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from server to storage."""
        vectorcore = VectorCore()

        # Simulate realistic AI workflow
        documents = [
            ("research_paper_1", [0.1, 0.2, 0.3, 0.4]),
            ("research_paper_2", [0.2, 0.3, 0.4, 0.5]),
            ("blog_post_1", [0.8, 0.7, 0.6, 0.5]),
            ("news_article_1", [0.9, 0.8, 0.7, 0.6]),
        ]

        # Add documents
        for doc_id, vector in documents:
            success = vectorcore.add_vector(doc_id, vector)
            self.assertTrue(success)

        # Query for similar content
        query_vector = [0.15, 0.25, 0.35, 0.45]  # Similar to research papers
        results = vectorcore.query_similar(query_vector, k=2)

        # Verify correct similarity ranking
        self.assertEqual(len(results), 2)
        self.assertIn("research_paper_1", results)
        self.assertIn("research_paper_2", results)

        # Test persistence
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            vectorcore.save_to_disk(tmp_path)

            # Load into new instance
            new_vectorcore = VectorCore()
            new_vectorcore.load_from_disk(tmp_path)

            # Verify same results
            new_results = new_vectorcore.query_similar(query_vector, k=2)
            self.assertEqual(set(results), set(new_results))
        finally:
            os.unlink(tmp_path)


def run_performance_benchmarks():
    """Run comprehensive performance benchmarks."""
    print("🚀 Running VectorCore Performance Benchmarks")
    print("=" * 60)

    # Create test suite with only performance tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformance)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        # Run only performance benchmarks
        success = run_performance_benchmarks()
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        print("🧪 Running VectorCore Test Suite")
        print("=" * 50)
        unittest.main(verbosity=2)
