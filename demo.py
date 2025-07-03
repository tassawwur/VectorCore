#!/usr/bin/env python3
"""
VectorCore Demo - Advanced Usage Examples

This script demonstrates the full capabilities of the VectorCore vector database,
including realistic use cases for AI applications.
"""

import socket
import time
import json


class VectorCoreClient:
    """Simple client wrapper for VectorCore server"""

    def __init__(self, host="localhost", port=8888):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """Connect to VectorCore server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        # Read welcome message
        self.socket.recv(1024)

    def send_command(self, command):
        """Send command and return response"""
        self.socket.send(f"{command}\n".encode("utf-8"))
        response = self.socket.recv(4096).decode("utf-8")
        return response.strip()

    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()


def demo_text_similarity():
    """Demo: Text similarity search using word embeddings"""
    print("=" * 60)
    print("📚 DEMO 1: Text Similarity Search")
    print("=" * 60)

    client = VectorCoreClient()
    client.connect()

    # Clear any existing data first
    client.send_command("CLEAR")

    # Simulate word embeddings for different concepts
    documents = {
        "king": [0.8, 0.2, 0.1, 0.9, 0.3],
        "queen": [0.75, 0.25, 0.15, 0.85, 0.35],  # Similar to king
        "royal": [0.7, 0.3, 0.2, 0.8, 0.4],  # Similar to king/queen
        "car": [0.1, 0.9, 0.8, 0.2, 0.7],  # Different domain
        "automobile": [0.15, 0.85, 0.75, 0.25, 0.65],  # Similar to car
        "python": [0.3, 0.6, 0.9, 0.4, 0.1],  # Programming
        "coding": [0.25, 0.65, 0.85, 0.45, 0.15],  # Similar to python
    }

    print("Adding document vectors...")
    for doc_id, vector in documents.items():
        response = client.send_command(f"ADD {doc_id} {json.dumps(vector)}")
        print(f"  📄 {doc_id}: {response.split('(')[0]}")

    print("\n🔍 Searching for documents similar to 'king' [0.8, 0.2, 0.1, 0.9, 0.3]...")
    response = client.send_command(f"QUERY 3 {json.dumps(documents['king'])}")
    print(response)

    print("\n🔍 Searching for documents similar to 'car' [0.1, 0.9, 0.8, 0.2, 0.7]...")
    response = client.send_command(f"QUERY 3 {json.dumps(documents['car'])}")
    print(response)

    client.close()


def demo_image_similarity():
    """Demo: Image similarity using image embeddings"""
    print("\n" + "=" * 60)
    print("🖼️  DEMO 2: Image Similarity Search")
    print("=" * 60)

    client = VectorCoreClient()
    client.connect()

    # Clear previous data to avoid dimension conflicts
    client.send_command("CLEAR")

    # Simulate image embeddings for different categories
    images = {
        "cat_1": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],  # Cat features
        "cat_2": [0.85, 0.15, 0.75, 0.25, 0.65, 0.35],  # Similar cat
        "dog_1": [0.7, 0.3, 0.6, 0.4, 0.8, 0.2],  # Dog features
        "dog_2": [0.75, 0.25, 0.65, 0.35, 0.85, 0.15],  # Similar dog
        "car_1": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],  # Vehicle features
        "car_2": [0.15, 0.85, 0.25, 0.75, 0.35, 0.65],  # Similar car
        "house_1": [0.4, 0.6, 0.5, 0.5, 0.6, 0.4],  # Building features
    }

    print("Adding image embeddings...")
    for image_id, vector in images.items():
        response = client.send_command(f"ADD {image_id} {json.dumps(vector)}")
        print(f"  🖼️  {image_id}: {response.split('(')[0]}")

    print("\n🔍 Finding images similar to cat_1...")
    response = client.send_command(f"QUERY 3 {json.dumps(images['cat_1'])}")
    print(response)

    print("\n🔍 Finding images similar to car_1...")
    response = client.send_command(f"QUERY 3 {json.dumps(images['car_1'])}")
    print(response)

    client.close()


def demo_persistence():
    """Demo: Database persistence and loading"""
    print("\n" + "=" * 60)
    print("💾 DEMO 3: Database Persistence")
    print("=" * 60)

    client = VectorCoreClient()
    client.connect()

    # Clear any existing data first
    client.send_command("CLEAR")

    # Add some data
    print("Adding sample data...")
    sample_data = {
        "item_1": [0.1, 0.2, 0.3],
        "item_2": [0.4, 0.5, 0.6],
        "item_3": [0.7, 0.8, 0.9],
    }

    for item_id, vector in sample_data.items():
        client.send_command(f"ADD {item_id} {json.dumps(vector)}")

    # Save to disk
    print("\n💾 Saving database to disk...")
    response = client.send_command("SAVE demo_backup.pkl")
    print(f"  {response}")

    # Clear database
    print("\n🗑️  Clearing database...")
    response = client.send_command("CLEAR")
    print(f"  {response}")

    # Show empty stats
    print("\n📊 Database stats after clearing:")
    response = client.send_command("STATS")
    print(f"  {response}")

    # Load from disk
    print("\n📂 Loading database from disk...")
    response = client.send_command("LOAD demo_backup.pkl")
    print(f"  {response}")

    # Show restored stats
    print("\n📊 Database stats after loading:")
    response = client.send_command("STATS")
    print(f"  {response}")

    # Verify data is restored
    print("\n🔍 Testing query on restored data...")
    response = client.send_command(f"QUERY 2 {json.dumps([0.15, 0.25, 0.35])}")
    print(response)

    client.close()


def demo_performance():
    """Demo: Performance with larger datasets"""
    print("\n" + "=" * 60)
    print("⚡ DEMO 4: Performance Test")
    print("=" * 60)

    client = VectorCoreClient()
    client.connect()

    # Clear any existing data
    client.send_command("CLEAR")

    print("Adding 100 random vectors for performance testing...")
    import random

    start_time = time.time()
    for i in range(100):
        # Generate random 3-dimensional vector (consistent with other demos)
        vector = [random.random() for _ in range(3)]
        client.send_command(f"ADD doc_{i:03d} {json.dumps(vector)}")

        if (i + 1) % 20 == 0:
            print(f"  ✅ Added {i + 1} vectors...")

    add_time = time.time() - start_time
    print(f"📊 Added 100 vectors in {add_time:.3f} seconds")

    # Test query performance
    print("\n🔍 Testing query performance...")
    query_vector = [0.5, 0.5, 0.5]  # 3-dimensional to match

    start_time = time.time()
    for _ in range(10):
        client.send_command(f"QUERY 10 {json.dumps(query_vector)}")
    query_time = time.time() - start_time

    print(f"📊 Performed 10 queries in {query_time:.3f} seconds")
    print(f"📊 Average query time: {query_time/10:.4f} seconds")

    # Show final stats
    print("\n📊 Final database statistics:")
    response = client.send_command("STATS")
    print(response)

    client.close()


def main():
    """Run all demonstrations"""
    print("🎯 VectorCore Advanced Demonstration")
    print("🚀 Showcasing high-performance vector similarity search")

    try:
        demo_text_similarity()
        demo_image_similarity()
        demo_persistence()
        demo_performance()

        print("\n" + "=" * 60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("🎉 VectorCore is ready for production use!")
        print("💡 Use cases: Search engines, recommendation systems, AI chatbots")

    except ConnectionRefusedError:
        print("❌ Cannot connect to VectorCore server.")
        print("💡 Please start the server first: python main.py")
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
