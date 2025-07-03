# VectorCore - A Vector Similarity Search Database

[![CI/CD Pipeline](https://github.com/tassawwur/VectorCore/actions/workflows/ci.yml/badge.svg)](https://github.com/tassawwur/VectorCore/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

VectorCore is an educational vector database implementation designed to demonstrate fundamental concepts in vector similarity search. Built from scratch in Python, it showcases how k-d trees can be used for efficient nearest neighbor search in multi-dimensional vector spaces.

## ��� Table of Contents

- [What is VectorCore?](#what-is-vectorcore)
- [The Problem it Addresses](#the-problem-it-addresses)
- [How it Works - Detailed Workflow](#how-it-works---detailed-workflow)
- [Technical Architecture](#technical-architecture)
- [Installation & Quick Start](#installation--quick-start)
- [Complete Usage Guide](#complete-usage-guide)
- [Use Cases & Examples](#use-cases--examples)
- [Performance Characteristics](#performance-characteristics)
- [Limitations & Scope](#limitations--scope)
- [API Reference](#api-reference)
- [Development & Testing](#development--testing)

## What is VectorCore?

VectorCore is a specialized in-memory database designed for storing and searching high-dimensional vectors. It implements a k-d tree data structure to enable efficient similarity search operations, which are fundamental to many AI and machine learning applications.

**Key Features:**
- **K-D Tree Indexing**: Logarithmic time complexity for nearest neighbor search
- **Multi-threaded Server**: TCP-based server supporting concurrent connections
- **Thread-Safe Operations**: Proper locking mechanisms for concurrent access
- **Persistence**: Save and load database state to/from disk
- **Memory Efficient**: Optimized data structures for vector storage
- **Educational Focus**: Clear, well-documented code for learning purposes

## The Problem it Addresses

Modern AI systems frequently need to find similar items by comparing their vector representations:

- **Text Similarity**: "renewable energy" and "sustainable power" might have similar embeddings
- **Image Recognition**: Photos of similar objects have similar feature vectors
- **Recommendation Systems**: Users with similar preferences have similar behavioral vectors
- **Content Discovery**: Similar documents cluster together in vector space

**The Challenge**: Finding similar vectors among large collections traditionally requires comparing each query against every stored vector (O(n) complexity), which becomes inefficient as datasets grow.

**VectorCore's Approach**: Uses k-d tree spatial partitioning to reduce average search complexity to O(log n), making similarity search more efficient for moderately-sized datasets.

## How it Works - Detailed Workflow

### 1. **System Initialization**

When VectorCore starts, several components are initialized:

```python
# Core database initialization
vectorcore = VectorCore()           # Main database instance
server = VectorCoreServer()         # TCP server for client connections
```

**What happens internally:**
- Memory allocation for vector storage dictionary
- Thread lock initialization for concurrent access
- K-d tree remains uninitialized until first vector insertion
- Database dimension is auto-detected from the first vector

### 2. **Vector Insertion Process**

When you add a vector, here's the complete workflow:

```python
# Client command: ADD doc_1 [0.1, 0.2, 0.3]
```

**Step-by-step process:**

1. **Input Parsing**:
   ```python
   command_parts = line.split(' ', 2)  # ['ADD', 'doc_1', '[0.1, 0.2, 0.3]']
   doc_id = command_parts[1]           # 'doc_1'
   vector_data = json.loads(command_parts[2])  # [0.1, 0.2, 0.3]
   ```

2. **Dimension Detection** (first vector only):
   ```python
   if self.dimension is None:
       self.dimension = len(vector_array)    # Set to 3
       self.kdtree = KDTree(self.dimension)  # Initialize k-d tree
   ```

3. **Validation**:
   ```python
   if len(vector_array) != self.dimension:
       raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {len(vector_array)}")
   ```

4. **Storage & Indexing**:
   ```python
   self.vectors[doc_id] = vector_array      # Store in dictionary for exact lookup
   self.kdtree.insert(vector_array, doc_id) # Insert into k-d tree for similarity search
   ```

5. **K-D Tree Insertion Logic**:
   ```python
   def _insert_recursive(self, node, vector, doc_id, depth):
       if node is None:
           return KDNode(vector, doc_id, depth % self.dimension)
       
       axis = depth % self.dimension  # Cycle through dimensions
       
       if vector[axis] < node.vector[axis]:
           node.left = self._insert_recursive(node.left, vector, doc_id, depth + 1)
       else:
           node.right = self._insert_recursive(node.right, vector, doc_id, depth + 1)
   ```

### 3. **Similarity Search Process**

When you query for similar vectors:

```python
# Client command: QUERY 5 [0.15, 0.25, 0.35]
```

**Detailed search workflow:**

1. **Query Preprocessing**:
   ```python
   k = int(command_parts[1])                    # Number of results: 5
   query_vector = json.loads(command_parts[2])  # [0.15, 0.25, 0.35]
   query_array = np.array(query_vector, dtype=np.float32)
   ```

2. **K-D Tree Search Algorithm**:
   ```python
   best_matches = []  # Max heap to track k closest points
   
   def search_recursive(node, depth=0):
       # Calculate Euclidean distance to current node
       distance = sqrt(sum((query_vector[i] - node.vector[i])**2 for i in range(dimension)))
       
       # Update best matches if this is closer
       if len(best_matches) < k:
           heapq.heappush(best_matches, (-distance, node.doc_id, distance))
       elif distance < best_matches[0][2]:  # Better than worst match
           heapq.heapreplace(best_matches, (-distance, node.doc_id, distance))
   ```

3. **Tree Traversal Optimization**:
   ```python
   axis = depth % self.dimension
   diff = query_vector[axis] - node.vector[axis]
   
   # Choose which subtree to search first
   if diff < 0:
       near_node, far_node = node.left, node.right
   else:
       near_node, far_node = node.right, node.left
   
   # Always search the "near" side
   search_recursive(near_node, depth + 1)
   
   # Only search "far" side if it might contain better matches
   if len(best_matches) < k or abs(diff) < best_matches[0][2]:
       search_recursive(far_node, depth + 1)
   ```

4. **Result Formatting**:
   ```python
   # Sort by distance and return document IDs
   results = [(doc_id, distance) for _, doc_id, distance in 
              sorted(best_matches, key=lambda x: x[2])]
   return [doc_id for doc_id, _ in results]
   ```

### 4. **Network Communication**

VectorCore uses a simple TCP protocol:

```python
# Server accepts connections and processes commands
def handle_client(self, client_socket, address):
    while True:
        data = client_socket.recv(1024).decode('utf-8').strip()
        response = self.process_command(data)
        client_socket.send(response.encode('utf-8'))
```

**Command Processing Flow**:
1. Parse incoming command string
2. Route to appropriate handler (ADD, QUERY, STATS, etc.)
3. Execute operation with proper error handling
4. Format response and send back to client

### 5. **Persistence Mechanism**

Database state can be saved and restored:

```python
# Save operation
data = {
    'dimension': self.dimension,
    'vectors': self.vectors,        # All vectors as dictionary
    'start_time': self.start_time
}
with open(filename, 'wb') as f:
    pickle.dump(data, f)

# Load operation
with open(filename, 'rb') as f:
    data = pickle.load(f)
    
self.dimension = data['dimension']
self.vectors = data['vectors']

# Rebuild k-d tree from saved vectors
self.kdtree = KDTree(self.dimension)
for doc_id, vector in self.vectors.items():
    self.kdtree.insert(vector, doc_id)
```

## Technical Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        VectorCore System                        │
├─────────────────────────────────────────────────────────────────┤
│  Client Layer                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────────┐ │
│  │   Python    │ │   Telnet    │ │      Custom TCP Client      │ │
│  │   Client    │ │   Client    │ │                             │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Network Layer                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               TCP Server (Multi-threaded)                   │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │ │
│  │  │ Connection  │ │  Command    │ │    Protocol Parser      │ │ │
│  │  │  Handler    │ │  Router     │ │                         │ │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Database Engine                                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    VectorCore Database                      │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │ │
│  │  │   Vector    │ │    K-D      │ │     Thread Safety       │ │ │
│  │  │  Storage    │ │    Tree     │ │       (RLock)           │ │ │
│  │  │             │ │   Index     │ │                         │ │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Storage Layer                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                     Persistence                             │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │ │
│  │  │   Pickle    │ │    Disk     │ │      State Recovery     │ │ │
│  │  │Serialization│ │   Storage   │ │                         │ │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Structures

**1. K-D Tree Node Structure:**
```python
class KDNode:
    def __init__(self, vector, doc_id, axis):
        self.vector = vector      # NumPy array of coordinates
        self.doc_id = doc_id      # Unique identifier string
        self.axis = axis          # Splitting dimension for this node
        self.left = None          # Left child (smaller values)
        self.right = None         # Right child (larger values)
```

**2. Vector Storage:**
```python
# Dual storage for different access patterns
self.vectors = {}           # Dict for O(1) exact lookup by doc_id
self.kdtree = KDTree()      # Tree for O(log n) similarity search
```

## Installation & Quick Start

### Prerequisites
- Python 3.9 or higher
- NumPy (automatically installed)

### Installation

```bash
# Clone the repository
git clone https://github.com/tassawwur/VectorCore.git
cd VectorCore

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

**1. Start the Server:**
```bash
python main.py
# Server starts on localhost:8888 by default
```

**2. Connect and Test:**
```bash
# Using telnet
telnet localhost 8888

# Add some vectors
ADD document_1 [0.1, 0.2, 0.3]
ADD document_2 [0.4, 0.5, 0.6]
ADD document_3 [0.2, 0.3, 0.4]

# Find similar vectors
QUERY 2 [0.15, 0.25, 0.35]

# Check statistics
STATS
```

**3. Run Demo:**
```bash
python demo.py
# Interactive demo showing various use cases
```

## Complete Usage Guide

### Command Reference

| Command | Syntax | Description | Example |
|---------|--------|-------------|---------|
| `ADD` | `ADD <doc_id> <vector>` | Store a vector with identifier | `ADD user_1 [0.1, 0.2, 0.3]` |
| `QUERY` | `QUERY <k> <vector>` | Find k nearest neighbors | `QUERY 5 [0.15, 0.25, 0.35]` |
| `GET` | `GET <doc_id>` | Retrieve stored vector | `GET user_1` |
| `REMOVE` | `REMOVE <doc_id>` | Delete a vector | `REMOVE user_1` |
| `STATS` | `STATS` | Database statistics | `STATS` |
| `SAVE` | `SAVE [filename]` | Save to disk | `SAVE my_vectors.pkl` |
| `LOAD` | `LOAD [filename]` | Load from disk | `LOAD my_vectors.pkl` |
| `CLEAR` | `CLEAR` | Remove all vectors | `CLEAR` |
| `HELP` | `HELP` | Show available commands | `HELP` |

### Python Client Example

```python
import socket
import json

class VectorCoreClient:
    def __init__(self, host='localhost', port=8888):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
    
    def add_vector(self, doc_id, vector):
        command = f"ADD {doc_id} {json.dumps(vector)}\n"
        self.socket.send(command.encode())
        return self.socket.recv(1024).decode().strip()
    
    def query_similar(self, vector, k=5):
        command = f"QUERY {k} {json.dumps(vector)}\n"
        self.socket.send(command.encode())
        return self.socket.recv(4096).decode().strip()
    
    def close(self):
        self.socket.close()

# Usage
client = VectorCoreClient()

# Add vectors
client.add_vector("product_1", [1.0, 2.0, 3.0])
client.add_vector("product_2", [1.1, 2.1, 3.1])
client.add_vector("product_3", [4.0, 5.0, 6.0])

# Find similar products
result = client.query_similar([1.05, 2.05, 3.05], k=2)
print(result)  # Should find product_1 and product_2 as most similar

client.close()
```

## Use Cases & Examples

### 1. **Document Similarity Search**

**Scenario**: Building a semantic search engine for research papers.

```python
# Add document embeddings (from BERT, Word2Vec, etc.)
ADD paper_123 [0.23, 0.45, 0.67, 0.12, 0.89]  # "Machine Learning in Healthcare"
ADD paper_456 [0.34, 0.56, 0.78, 0.23, 0.90]  # "AI for Medical Diagnosis"
ADD paper_789 [0.65, 0.43, 0.21, 0.87, 0.54]  # "Climate Change Data Analysis"

# User searches for "medical AI applications"
QUERY 5 [0.28, 0.51, 0.72, 0.18, 0.89]
# Expected: paper_456, paper_123 (medical/AI papers rank higher)
```

**How it works**: Text is converted to vectors using pre-trained models, then VectorCore finds documents with similar semantic meaning.

### 2. **Product Recommendation System**

**Scenario**: E-commerce site recommending similar products.

```python
# Product feature vectors (price, category, ratings, etc. encoded)
ADD laptop_001 [0.8, 0.6, 0.9, 0.7]  # High-end gaming laptop
ADD laptop_002 [0.7, 0.5, 0.8, 0.6]  # Mid-range gaming laptop  
ADD phone_001 [0.2, 0.9, 0.6, 0.8]   # Smartphone
ADD tablet_001 [0.4, 0.7, 0.5, 0.7]  # Tablet

# User viewed: mid-range gaming laptop
QUERY 3 [0.75, 0.55, 0.85, 0.65]
# Expected: laptop_002, laptop_001, tablet_001 (similar categories/features)
```

### 3. **Image Feature Search**

**Scenario**: Finding visually similar images using CNN features.

```python
# CNN feature vectors from last layer (e.g., ResNet-50)
ADD cat_image_1 [0.12, 0.88, 0.34, 0.67]   # Orange tabby cat
ADD cat_image_2 [0.15, 0.85, 0.38, 0.71]   # Orange Persian cat
ADD dog_image_1 [0.45, 0.23, 0.78, 0.56]   # Golden retriever
ADD car_image_1 [0.89, 0.12, 0.45, 0.23]   # Red sports car

# Search with new cat image
QUERY 2 [0.14, 0.86, 0.36, 0.69]
# Expected: cat_image_1, cat_image_2 (similar visual features)
```

### 4. **User Behavior Analysis**

**Scenario**: Finding users with similar preferences for collaborative filtering.

```python
# User preference vectors (genre preferences, activity patterns, etc.)
ADD user_001 [0.9, 0.1, 0.8, 0.3]  # Loves action, dislikes romance
ADD user_002 [0.8, 0.2, 0.7, 0.4]  # Similar to user_001
ADD user_003 [0.2, 0.9, 0.3, 0.8]  # Loves romance, dislikes action

# Find users similar to user_001 for recommendations
QUERY 5 [0.85, 0.15, 0.75, 0.35]
# Expected: user_002, user_001 (similar preferences)
```

## Performance Characteristics

### Time Complexity

| Operation | Average Case | Worst Case | Notes |
|-----------|--------------|------------|-------|
| **Insert** | O(log n) | O(n) | Worst case when tree becomes unbalanced |
| **Search** | O(log n) | O(n) | K-d tree degrades to linear scan in high dimensions |
| **Delete** | O(log n) | O(n) | Requires tree rebuilding in current implementation |
| **Range Query** | O(log n + m) | O(n) | m = number of results returned |

### Space Complexity

- **Memory Usage**: O(n × d) where n = number of vectors, d = vector dimension
- **Tree Overhead**: ~2× memory usage due to tree structure and vector storage
- **Typical**: ~50-100 bytes per vector for moderate dimensions (100-500D)

### Measured Performance

**Test Environment**: Intel i5-8400, 16GB RAM, Python 3.11

| Dataset Size | Dimension | Insert Rate | Query Rate | Memory Usage |
|--------------|-----------|-------------|------------|--------------|
| 1,000 vectors | 128D | ~1,600/sec | ~560/sec | ~2MB |
| 10,000 vectors | 128D | ~1,400/sec | ~420/sec | ~15MB |
| 100,000 vectors | 128D | ~1,200/sec | ~280/sec | ~140MB |

**Notes**: 
- Performance decreases with dataset size due to tree depth
- High-dimensional vectors (>20D) may see degraded k-d tree performance
- Results are from local testing and may vary by hardware

### Scalability Considerations

**Works Well For:**
- Small to medium datasets (< 1M vectors)
- Moderate dimensions (< 100D)
- Exact nearest neighbor search requirements
- Educational and prototyping purposes

**Limitations:**
- K-d trees become less efficient in high dimensions (>20-50D)
- Memory-bound for very large datasets
- Single-node architecture (no distributed support)
- Tree rebuilding required for deletions

## Limitations & Scope

### Current Limitations

**1. Dimensional Scalability**
- K-d trees suffer from "curse of dimensionality" 
- Performance degrades significantly above ~20-50 dimensions
- For high-dimensional data, consider LSH or learned indices

**2. Dataset Size**
- Memory-bound architecture limits scalability
- No built-in sharding or distributed support
- Optimal for datasets under 1M vectors

**3. Update Operations**
- Deletions require full tree rebuilding
- No incremental updates for optimal tree balance
- Better suited for append-mostly workloads

**4. Approximate Search**
- Only supports exact nearest neighbor search
- No approximate algorithms for faster queries
- May be slower than modern approximate methods (HNSW, LSH)

### Design Choices & Trade-offs

**Why K-D Trees?**
- **Educational Value**: Clear algorithm that's easy to understand
- **Exact Results**: Guaranteed to find true nearest neighbors
- **Balanced Performance**: Good average-case performance for moderate dimensions
- **Memory Efficiency**: Relatively low memory overhead

**What This Is NOT:**
- A production vector database (use Pinecone, Weaviate, or Qdrant)
- A high-performance system for billion-scale datasets
- An approximate search solution for high-dimensional vectors
- A distributed or cloud-native system

### When to Use VectorCore

**Good For:**
- Learning about vector databases and similarity search
- Prototyping vector-based applications
- Educational projects and algorithm study
- Small-scale applications with exact search requirements
- Understanding k-d tree implementations

**Consider Alternatives For:**
- Production applications requiring high availability
- Large-scale datasets (>1M vectors)
- High-dimensional vectors (>100D)
- Applications requiring sub-millisecond latency
- Distributed or cloud-native deployments

## API Reference

### TCP Protocol

VectorCore uses a line-based text protocol over TCP:

```
Client Request:  COMMAND ARGS\n
Server Response: STATUS_MESSAGE\n
```

### Command Details

#### ADD Command
```
Syntax: ADD <doc_id> <vector_json>
Example: ADD document_1 [0.1, 0.2, 0.3, 0.4]
Success: ✅ Added vector for doc_id 'document_1'
Error: ❌ Error adding vector for document_1: <error_message>
```

#### QUERY Command
```
Syntax: QUERY <k> <vector_json>
Example: QUERY 5 [0.15, 0.25, 0.35, 0.45]
Success: ��� Found 5 similar vectors: ['doc_2', 'doc_1', 'doc_3', ...]
Error: ❌ Error querying similar vectors: <error_message>
```

#### GET Command
```
Syntax: GET <doc_id>
Example: GET document_1
Success: ��� Vector for 'document_1': [0.1, 0.2, 0.3, 0.4]
Error: ❌ Vector not found for doc_id: document_1
```

#### STATS Command
```
Syntax: STATS
Example: STATS
Response: ��� VectorCore Stats:
         • Vectors: 1000
         • Dimension: 128
         • Uptime: 3600.5 seconds
         • Memory usage: 1000 vectors
```

### Error Handling

**Common Error Cases:**
- **Dimension Mismatch**: Vector dimension doesn't match database
- **Invalid JSON**: Malformed vector data
- **Missing Document**: Requested doc_id not found
- **Invalid Command**: Unknown command or syntax error
- **Memory Errors**: Out of memory during large operations

**Error Response Format:**
```
❌ Error: <detailed_error_message>
```

## Development & Testing

### Code Structure

```
VectorCore/
├── kdtree.py          # K-D tree implementation
├── vectorcore.py      # Main database logic
├── server.py          # TCP server implementation
├── main.py            # CLI entry point
├── tests.py           # Comprehensive test suite
├── demo.py            # Interactive demonstrations
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container configuration
├── .github/
│   └── workflows/
│       └── ci.yml     # CI/CD pipeline
└── README.md          # This documentation
```

### Running Tests

```bash
# Run all tests
python -m unittest tests.py

# Run specific test categories
python -m unittest tests.TestKDTree      # K-D tree tests
python -m unittest tests.TestVectorCore  # Database tests
python -m unittest tests.TestPerformance # Performance benchmarks

# Run with verbose output
python -m unittest tests.py -v
```

### Code Quality Checks

```bash
# Type checking
python -m mypy *.py --ignore-missing-imports

# Code formatting
python -m black . --check

# Linting
python -m flake8 . --max-line-length=88

# Security scanning
python -m bandit -r .
```

### Contributing

This project is primarily educational, but improvements are welcome:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/improvement`
3. **Make changes with tests**: Ensure all tests pass
4. **Submit a pull request**: With clear description of changes

**Areas for Enhancement:**
- Better tree balancing algorithms
- Additional distance metrics (cosine, Manhattan)
- Performance optimizations
- More comprehensive error handling
- Additional client language bindings

### Docker Usage

```bash
# Build image
docker build -t vectorcore .

# Run container
docker run -p 8888:8888 vectorcore

# Run with custom port
docker run -p 9999:8888 vectorcore
```

## Educational Value

VectorCore demonstrates several important computer science concepts:

### Data Structures & Algorithms
- **K-D Trees**: Spatial data structures and tree algorithms
- **Heap Operations**: Priority queues for k-nearest neighbors
- **Tree Traversal**: Recursive algorithms and pruning strategies

### Software Engineering
- **Concurrent Programming**: Thread-safe operations and locking
- **Network Programming**: TCP servers and protocol design
- **Error Handling**: Robust exception handling and validation
- **Testing**: Unit tests, integration tests, and benchmarking

### System Design
- **Database Design**: Storage and indexing strategies
- **API Design**: Simple and intuitive command interfaces
- **Performance Optimization**: Algorithm and data structure choices
- **Persistence**: Serialization and data recovery

This implementation serves as a practical example of how foundational algorithms and software engineering principles come together to solve real-world problems in AI and data science.

---

**License**: MIT License - See [LICENSE](LICENSE) for details

**Author**: Built as an educational project to demonstrate vector database concepts and implementation techniques.
