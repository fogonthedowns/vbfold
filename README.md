# VBfold Rust

A high-performance vector database implementation in Rust, designed as a compatible alternative to TurboPuffer with significant performance improvements for concurrent workloads.

## Overview

VBfold Rust is a memory-based vector database that provides fast similarity search capabilities for high-dimensional vectors. Built from the ground up in Rust, it addresses concurrency issues found in other implementations while delivering exceptional performance for both ingestion and query operations.

## Performance Characteristics

### Insert Performance
- **2.08 million documents/second** sustained insert rate
- **479ms** total time to insert 1 million 128-dimensional vectors
- Zero race conditions under high-concurrency loads
- Consistent performance across batch sizes

### Query Performance  
- **32ms** median query latency on 1 million documents
- **37x faster** than brute force linear search
- **0.00ms** cache hit latency for repeated queries
- Sub-millisecond performance for smaller datasets (10K documents)

### Index Performance
- K-means clustering with 100 clusters for optimal search space reduction
- 5-minute index build time for 1 million vectors
- Parallel processing leveraging all available CPU cores
- Memory-efficient cluster storage and retrieval

## Architecture

### Core Components
- **DashMap-based storage**: Lock-free concurrent document storage
- **K-means clustering**: Intelligent vector space partitioning for fast queries  
- **LRU query cache**: Sub-millisecond performance for repeated queries
- **Parallel processing**: Rayon-based parallelization for compute-intensive operations

### API Compatibility
- RESTful HTTP API compatible with VBfold clients
- JSON-based request/response format
- Support for vector, text, and hybrid search modes
- Metadata filtering capabilities

## Quick Start

### Prerequisites
- Rust 1.70+ 
- 8GB+ RAM recommended for million-scale datasets

### Installation and Running
```bash
git clone <repository-url>
cd vbfold-rust
cargo run --release
```

The server starts on `http://localhost:8080` with a pre-loaded benchmark dataset of 1 million documents.

### API Usage

#### Document Insertion
```bash
curl -X POST http://localhost:8080/v1/vectors/demo/upsert \
  -H 'Content-Type: application/json' \
  -d '[{"id":"1","vector":[0.1,0.2,0.3,0.4],"text":"sample document"}]'
```

#### Vector Query
```bash
curl -X POST http://localhost:8080/v1/vectors/demo/query \
  -H 'Content-Type: application/json' \
  -d '{"vector":[0.1,0.2,0.3,0.4],"top_k":10}'
```

#### Namespace Information
```bash
curl http://localhost:8080/v1/vectors/demo
```

## Benchmarks

Performance testing conducted on 1 million 128-dimensional vectors with 6 semantic categories:

| Operation | Performance | Notes |
|-----------|-------------|--------|
| Document insertion | 2.08M docs/sec | Sustained rate with batching |
| Cold query | 30.94ms | First query on dataset |
| Warm query (P50) | 32.65ms | Median performance |
| Warm query (P95) | 41.32ms | 95th percentile |
| Cache hit | 0.00ms | LRU cache performance |
| Index build | 304 seconds | K-means clustering |

## Technical Details

### Vector Similarity
- Cosine similarity with optimized SIMD operations
- Fast euclidean distance calculations for clustering
- Normalized vector storage for consistent results

### Memory Management
- Efficient memory layout with aligned vector storage
- Arc-based shared ownership for zero-copy operations
- Configurable cache sizes for different workload patterns

### Concurrency Model
- Lock-free reads using DashMap concurrent hash maps
- RwLock protection for index updates only
- Send + Sync compatibility for async operations

## Configuration

Key configuration options in the source code:

```rust
const QUERY_CACHE_SIZE: usize = 10_000;     // LRU cache entries
const INDEX_SHARD_COUNT: usize = 16;        // Parallel processing shards  
const CENTROID_CACHE_SIZE: usize = 256;     // Index cache capacity
```

## Limitations

- Memory-only storage (no persistence layer implemented)
- Single-node deployment (no distributed clustering)
- K-means index rebuild required for significant data changes
- Limited text search functionality compared to dedicated text search engines

## Comparison with VBfold

| Metric | VBfold Rust | TurboPuffer Cloud |
|--------|------------------|-------------------|
| Cold query (1M docs) | 31ms | 402ms |
| Warm query (1M docs) | 33ms | 16ms |
| Insert rate | 2.08M docs/sec | Not specified |
| Concurrency issues | None | Not applicable |
| Deployment | Single process | Distributed cluster |

## Future Improvements

- Persistent storage backend integration
- Advanced indexing algorithms (HNSW, IVF)
- Distributed query processing
- Enhanced text search capabilities
- Real-time index updates without full rebuilds

## License

[Specify license here]

## Contributing

[Contribution guidelines here]
