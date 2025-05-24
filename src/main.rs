use ahash::AHashMap;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use crossbeam::channel::{bounded, Sender};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tracing::{info, warn};

// Constants optimized for performance
const VECTOR_CHUNK_SIZE: usize = 64;
const CENTROID_CACHE_SIZE: usize = 256;
const POSTING_LIST_BLOCK_SIZE: usize = 128;
const QUERY_CACHE_SIZE: usize = 10_000;
const INDEX_SHARD_COUNT: usize = 16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
fn fast_cosine_similarity_raw(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let base = i * 4;
        
        dot += a[base] * b[base] + a[base+1] * b[base+1] + 
               a[base+2] * b[base+2] + a[base+3] * b[base+3];
        
        norm_a += a[base] * a[base] + a[base+1] * a[base+1] + 
                  a[base+2] * a[base+2] + a[base+3] * a[base+3];
        
        norm_b += b[base] * b[base] + b[base+1] * b[base+1] + 
                  b[base+2] * b[base+2] + b[base+3] * b[base+3];
    }
    
    for i in (chunks * 4)..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

#[derive(Debug)]
struct ScoredDoc {
    score: f32,
    id: String,
    text: Option<String>,
    metadata: Option<serde_json::Value>,
}

impl PartialEq for ScoredDoc {
    fn eq(&self, other: &Self) -> bool { 
        self.score == other.score 
    }
}

impl Eq for ScoredDoc {}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[derive(Clone)]
struct SimpleCluster {
    centroid: Vec<f32>,
    doc_ids: Vec<String>,
    centroid_norm: f32,
}

struct SimplifiedIndex {
    clusters: Vec<SimpleCluster>,
    cluster_count: usize,
}

impl SimplifiedIndex {
    fn new(cluster_count: usize) -> Self {
        Self {
            clusters: Vec::with_capacity(cluster_count),
            cluster_count,
        }
    }
    
    fn build_index(&mut self, documents: &DashMap<String, Arc<Document>>) {
        let vectors_with_ids: Vec<(String, Vec<f32>)> = documents
            .iter()
            .filter_map(|entry| {
                let doc = entry.value();
                doc.vector.as_ref().map(|v| (doc.id.clone(), v.clone()))
            })
            .collect();
        
        if vectors_with_ids.is_empty() {
            return;
        }
        
        let dim = vectors_with_ids[0].1.len();
        self.clusters = self.kmeans_clustering(&vectors_with_ids, dim);
        
        info!("📊 Built index with {} clusters covering {} vectors", 
              self.clusters.len(), vectors_with_ids.len());
    }
    
    fn kmeans_clustering(&self, vectors_with_ids: &[(String, Vec<f32>)], dim: usize) -> Vec<SimpleCluster> {
        info!("🔧 Starting k-means with {} vectors and {} clusters...", vectors_with_ids.len(), self.cluster_count);
        
        let k = self.cluster_count.min(vectors_with_ids.len()).max(1);
        
        // Initialize centroids by sampling every k-th vector
        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|i| vectors_with_ids[i * vectors_with_ids.len() / k].1.clone())
            .collect();
        
        // Only do 2 iterations instead of 3 for speed
        for iteration in 0..2 {
            info!("📊 K-means iteration {}/2...", iteration + 1);
            
            // Parallel assignment of vectors to clusters
            let assignments: Vec<usize> = vectors_with_ids
                .par_iter()
                .map(|(_, vector)| {
                    let mut best_cluster = 0;
                    let mut best_distance = f32::INFINITY;
                    
                    for (cluster_idx, centroid) in centroids.iter().enumerate() {
                        let distance = euclidean_distance(vector, centroid);
                        if distance < best_distance {
                            best_distance = distance;
                            best_cluster = cluster_idx;
                        }
                    }
                    best_cluster
                })
                .collect();
            
            // Update centroids in parallel
            centroids = (0..k).into_par_iter().map(|cluster_id| {
                let mut new_centroid = vec![0.0; dim];
                let mut count = 0;
                
                for (idx, &assigned_cluster) in assignments.iter().enumerate() {
                    if assigned_cluster == cluster_id {
                        for (i, &val) in vectors_with_ids[idx].1.iter().enumerate() {
                            new_centroid[i] += val;
                        }
                        count += 1;
                    }
                }
                
                if count > 0 {
                    for val in &mut new_centroid {
                        *val /= count as f32;
                    }
                }
                new_centroid
            }).collect();
        }
        
        // Final assignment with progress reporting
        info!("🔗 Final assignment phase...");
        let mut final_clusters = Vec::new();
        for (cluster_idx, centroid) in centroids.iter().enumerate() {
            let mut cluster_doc_ids = Vec::new();
            
            for (doc_id, vector) in vectors_with_ids.iter() {
                let mut best_cluster = 0;
                let mut best_distance = f32::INFINITY;
                
                for (c_idx, c_centroid) in centroids.iter().enumerate() {
                    let distance = euclidean_distance(vector, c_centroid);
                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = c_idx;
                    }
                }
                
                if best_cluster == cluster_idx {
                    cluster_doc_ids.push(doc_id.clone());
                }
            }
            
            if !cluster_doc_ids.is_empty() {
                final_clusters.push(SimpleCluster {
                    centroid: centroid.clone(),
                    doc_ids: cluster_doc_ids,
                    centroid_norm: centroid.iter().map(|x| x * x).sum::<f32>().sqrt(),
                });
            }
        }
        
        final_clusters
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<serde_json::Value>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub include_text: bool,
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub id: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    pub results: Vec<QueryResult>,
    pub latency_ms: f64,
    pub cache_hit: bool,
    pub roundtrips: u32,
}

#[derive(Clone)]
struct AlignedVector {
    data: Vec<f32>,
    dim: usize,
}

impl AlignedVector {
    fn new(vec: Vec<f32>) -> Self {
        let dim = vec.len();
        Self { data: vec, dim }
    }

    #[inline(always)]
    fn dot_product_simd(&self, other: &[f32]) -> f32 {
        self.data.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }

    #[inline(always)]
    fn cosine_similarity(&self, other: &[f32]) -> f32 {
        let dot = self.dot_product_simd(other);
        let norm_self = self.norm();
        let norm_other = other.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_self == 0.0 || norm_other == 0.0 {
            0.0
        } else {
            dot / (norm_self * norm_other)
        }
    }

    #[inline(always)]
    fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

struct SPFreshIndex {
    centroids: Vec<Centroid>,
    clusters: Vec<Cluster>,
    centroid_norms: Vec<f32>,
    cluster_index: Vec<Vec<usize>>,
}

struct Centroid {
    id: usize,
    vector: AlignedVector,
    count: AtomicU64,
}

struct Cluster {
    centroid_id: usize,
    doc_ids: roaring::RoaringBitmap,
    vectors: Vec<AlignedVector>,
}

struct BM25Index {
    terms: DashMap<String, CompressedPostingList>,
    doc_lengths: DashMap<usize, u32>,
    avg_doc_len: AtomicU64,
    num_docs: AtomicU64,
}

struct CompressedPostingList {
    docs: roaring::RoaringBitmap,
    frequencies: Vec<u8>,
}

struct TurboCache {
    memory_cache: Arc<DashMap<String, CachedData>>,
    query_cache: Arc<Mutex<lru::LruCache<u64, Arc<Vec<QueryResult>>>>>,
    stats: CacheStats,
}

#[derive(Clone)]
struct CachedData {
    data: Bytes,
    last_access: Instant,
    access_count: Arc<AtomicU64>,
}

struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

struct Namespace {
    name: String,
    documents: Arc<DashMap<String, Arc<Document>>>,
    simple_index: Arc<RwLock<SimplifiedIndex>>,
    vector_index: Arc<RwLock<SPFreshIndex>>,
    text_index: Arc<RwLock<BM25Index>>,
    metadata_index: Arc<RwLock<MetadataIndex>>,
    wal: Arc<RwLock<Vec<WALEntry>>>,
    write_buffer: Arc<Mutex<Vec<Document>>>,
    indexing_active: AtomicBool,
    last_indexed: AtomicU64,
}

struct MetadataIndex {
    fields: DashMap<String, DashMap<serde_json::Value, roaring::RoaringBitmap>>,
}

#[derive(Clone, Serialize, Deserialize)]
struct WALEntry {
    timestamp: u64,
    documents: Vec<Document>,
    batch_id: String,
}

pub struct TurboPuffer {
    namespaces: Arc<DashMap<String, Arc<Namespace>>>,
    object_storage: Arc<ObjectStorage>,
    cache: Arc<TurboCache>,
    write_scheduler: Arc<WriteScheduler>,
    query_executor: Arc<QueryExecutor>,
}

struct ObjectStorage {
    data: DashMap<String, Bytes>,
    latency: Duration,
}

struct WriteScheduler {
    sender: Sender<WriteBatch>,
    batch_size: usize,
    flush_interval: Duration,
}

struct WriteBatch {
    namespace: String,
    documents: Vec<Document>,
}

struct QueryExecutor {
    thread_pool: rayon::ThreadPool,
    shard_count: usize,
}

impl TurboPuffer {
    pub fn new() -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .build()
            .unwrap();

        let (tx, rx) = bounded(1000);
        
        let object_storage = Arc::new(ObjectStorage {
            data: DashMap::new(),
            latency: Duration::from_millis(100),
        });
        
        tokio::spawn(async move {
            while let Ok(_batch) = rx.recv() {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });

        Self {
            namespaces: Arc::new(DashMap::new()),
            object_storage,
            cache: Arc::new(TurboCache::new()),
            write_scheduler: Arc::new(WriteScheduler {
                sender: tx,
                batch_size: 1000,
                flush_interval: Duration::from_millis(100),
            }),
            query_executor: Arc::new(QueryExecutor {
                thread_pool,
                shard_count: INDEX_SHARD_COUNT,
            }),
        }
    }

    pub async fn upsert(&self, namespace: &str, documents: Vec<Document>) -> Result<(), String> {
        let ns = self.get_or_create_namespace(namespace).await;
        
        self.write_scheduler
            .sender
            .send(WriteBatch {
                namespace: namespace.to_string(),
                documents: documents.clone(),
            })
            .map_err(|e| e.to_string())?;

        for doc in documents {
            ns.documents.insert(doc.id.clone(), Arc::new(doc));
        }

        Ok(())
    }

    pub async fn query(&self, namespace: &str, request: QueryRequest) -> Result<QueryResponse, String> {
        let start = Instant::now();
        let ns = self.get_or_create_namespace(namespace).await;
        
        let cache_key = self.compute_query_cache_key(&request);
        if let Some(cached) = self.cache.get_query_result(cache_key) {
            return Ok(QueryResponse {
                results: cached.to_vec(),
                latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                cache_hit: true,
                roundtrips: 0,
            });
        }

        let results = if let Some(vector) = &request.vector {
            if let Some(text) = &request.text {
                self.hybrid_search(&ns, vector, text, request.top_k).await?
            } else {
                self.vector_search(&ns, vector, request.top_k).await?
            }
        } else if let Some(text) = &request.text {
            self.text_search(&ns, text, request.top_k).await?
        } else {
            return Err("No query vector or text provided".to_string());
        };

        let filtered_results = if let Some(filters) = &request.filters {
            self.apply_filters(results, filters, &ns).await?
        } else {
            results
        };

        self.cache.put_query_result(cache_key, Arc::new(filtered_results.clone()));

        Ok(QueryResponse {
            results: filtered_results,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            cache_hit: false,
            roundtrips: 1,
        })
    }

    async fn vector_search(&self, ns: &Arc<Namespace>, query: &[f32], top_k: usize) -> Result<Vec<QueryResult>, String> {
        // Check if index is empty first, then get clusters without holding lock across await
        let is_empty = {
            let index = ns.simple_index.read();
            index.clusters.is_empty()
        };
        
        if is_empty {
            return self.brute_force_search(ns, query, top_k).await;
        }
        
        // Clone the clusters data we need to avoid holding the lock
        let clusters = {
            let index = ns.simple_index.read();
            index.clusters.clone()
        };
        
        let mut cluster_scores: Vec<(usize, f32)> = clusters
            .iter()
            .enumerate()
            .map(|(idx, cluster)| {
                let score = fast_cosine_similarity_raw(query, &cluster.centroid);
                (idx, score)
            })
            .collect();
        
        cluster_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let search_clusters = cluster_scores.len().min(3);
        let mut heap = std::collections::BinaryHeap::with_capacity(top_k + 1);
        
        for i in 0..search_clusters {
            let cluster_idx = cluster_scores[i].0;
            let cluster = &clusters[cluster_idx];
            
            for doc_id in &cluster.doc_ids {
                if let Some(doc) = ns.documents.get(doc_id) {
                    if let Some(vector) = &doc.vector {
                        let score = fast_cosine_similarity_raw(query, vector);
                        
                        if heap.len() < top_k {
                            heap.push(ScoredDoc { 
                                score, 
                                id: doc.id.clone(),
                                text: doc.text.clone(),
                                metadata: doc.metadata.clone(),
                            });
                        } else if score > heap.peek().unwrap().score {
                            heap.pop();
                            heap.push(ScoredDoc { 
                                score, 
                                id: doc.id.clone(),
                                text: doc.text.clone(),
                                metadata: doc.metadata.clone(),
                            });
                        }
                    }
                }
            }
        }
        
        let mut results: Vec<QueryResult> = heap.into_iter().map(|scored| {
            QueryResult {
                id: scored.id,
                score: scored.score,
                vector: None,
                text: scored.text,
                metadata: scored.metadata,
            }
        }).collect();
        
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(results)
    }

    async fn brute_force_search(&self, ns: &Arc<Namespace>, query: &[f32], top_k: usize) -> Result<Vec<QueryResult>, String> {
        let mut heap = std::collections::BinaryHeap::with_capacity(top_k + 1);
        
        for entry in ns.documents.iter() {
            let doc = entry.value();
            if let Some(vector) = &doc.vector {
                let score = fast_cosine_similarity_raw(query, vector);
                
                if heap.len() < top_k {
                    heap.push(ScoredDoc { 
                        score, 
                        id: doc.id.clone(),
                        text: doc.text.clone(),
                        metadata: doc.metadata.clone(),
                    });
                } else if score > heap.peek().unwrap().score {
                    heap.pop();
                    heap.push(ScoredDoc { 
                        score, 
                        id: doc.id.clone(),
                        text: doc.text.clone(),
                        metadata: doc.metadata.clone(),
                    });
                }
            }
        }
        
        let mut results: Vec<QueryResult> = heap.into_iter().map(|scored| {
            QueryResult {
                id: scored.id,
                score: scored.score,
                vector: None,
                text: scored.text,
                metadata: scored.metadata,
            }
        }).collect();
        
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(results)
    }

    async fn text_search(&self, ns: &Arc<Namespace>, query: &str, top_k: usize) -> Result<Vec<QueryResult>, String> {
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let mut results = Vec::new();
        
        for entry in ns.documents.iter() {
            let doc = entry.value();
            if let Some(text) = &doc.text {
                let text_lower = text.to_lowercase();
                let mut score = 0.0;
                
                for term in &query_terms {
                    if text_lower.contains(&term.to_lowercase()) {
                        score += 1.0;
                    }
                }
                
                if score > 0.0 {
                    results.push(QueryResult {
                        id: doc.id.clone(),
                        score,
                        vector: None,
                        text: doc.text.clone(),
                        metadata: doc.metadata.clone(),
                    });
                }
            }
        }
        
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        
        Ok(results)
    }

    async fn hybrid_search(&self, ns: &Arc<Namespace>, vector: &[f32], text: &str, top_k: usize) -> Result<Vec<QueryResult>, String> {
        let (vector_results, text_results) = tokio::join!(
            self.vector_search(ns, vector, top_k * 2),
            self.text_search(ns, text, top_k * 2)
        );
        
        let vector_results = vector_results?;
        let text_results = text_results?;
        
        let mut combined: AHashMap<String, QueryResult> = AHashMap::new();
        
        for result in vector_results {
            combined.insert(result.id.clone(), QueryResult {
                score: result.score * 0.5,
                ..result
            });
        }
        
        for result in text_results {
            combined
                .entry(result.id.clone())
                .and_modify(|r| r.score += result.score * 0.5)
                .or_insert(QueryResult {
                    score: result.score * 0.5,
                    ..result
                });
        }
        
        let mut results: Vec<QueryResult> = combined.into_values().collect();
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        
        Ok(results)
    }

    async fn apply_filters(&self, results: Vec<QueryResult>, filters: &serde_json::Value, ns: &Arc<Namespace>) -> Result<Vec<QueryResult>, String> {
        // We don't actually use the metadata index yet, so let's not hold the lock
        // let _metadata_index = ns.metadata_index.read();
        
        Ok(results
            .into_iter()
            .filter(|result| {
                if let Some(doc) = ns.documents.get(&result.id) {
                    if let Some(metadata) = &doc.metadata {
                        if let Some(filter_obj) = filters.as_object() {
                            for (key, value) in filter_obj {
                                if let Some(doc_value) = metadata.get(key) {
                                    if doc_value != value {
                                        return false;
                                    }
                                } else {
                                    return false;
                                }
                            }
                        }
                    }
                }
                true
            })
            .collect())
    }

    async fn get_or_create_namespace(&self, name: &str) -> Arc<Namespace> {
        if let Some(ns) = self.namespaces.get(name) {
            return ns.clone();
        }

        let ns = Arc::new(Namespace::new(name.to_string()));
        self.namespaces.insert(name.to_string(), ns.clone());
        
        let _ns_clone = ns.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });
        
        ns
    }

    fn compute_query_cache_key(&self, request: &QueryRequest) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = ahash::AHasher::default();
        
        if let Some(vector) = &request.vector {
            for v in vector {
                v.to_bits().hash(&mut hasher);
            }
        }
        
        if let Some(text) = &request.text {
            text.hash(&mut hasher);
        }
        
        request.top_k.hash(&mut hasher);
        hasher.finish()
    }
}

impl Namespace {
    fn new(name: String) -> Self {
        let vector_index = SPFreshIndex {
            centroids: Vec::with_capacity(CENTROID_CACHE_SIZE),
            clusters: Vec::with_capacity(CENTROID_CACHE_SIZE),
            centroid_norms: Vec::with_capacity(CENTROID_CACHE_SIZE),
            cluster_index: Vec::with_capacity(CENTROID_CACHE_SIZE),
        };

        Self {
            name,
            documents: Arc::new(DashMap::with_capacity(1_000_000)),
            simple_index: Arc::new(RwLock::new(SimplifiedIndex::new(100))),
            vector_index: Arc::new(RwLock::new(vector_index)),
            text_index: Arc::new(RwLock::new(BM25Index {
                terms: DashMap::new(),
                doc_lengths: DashMap::new(),
                avg_doc_len: AtomicU64::new(0),
                num_docs: AtomicU64::new(0),
            })),
            metadata_index: Arc::new(RwLock::new(MetadataIndex {
                fields: DashMap::new(),
            })),
            wal: Arc::new(RwLock::new(Vec::new())),
            write_buffer: Arc::new(Mutex::new(Vec::with_capacity(10_000))),
            indexing_active: AtomicBool::new(false),
            last_indexed: AtomicU64::new(0),
        }
    }
}

impl TurboCache {
    fn new() -> Self {
        Self {
            memory_cache: Arc::new(DashMap::with_capacity(10_000)),
            query_cache: Arc::new(Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::new(QUERY_CACHE_SIZE).unwrap()))),
            stats: CacheStats {
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                evictions: AtomicU64::new(0),
            },
        }
    }

    fn get_query_result(&self, key: u64) -> Option<Arc<Vec<QueryResult>>> {
        let mut cache = self.query_cache.lock();
        if let Some(result) = cache.get(&key) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            Some(result.clone())
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    fn put_query_result(&self, key: u64, results: Arc<Vec<QueryResult>>) {
        let mut cache = self.query_cache.lock();
        if cache.put(key, results).is_some() {
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub async fn create_app(turbo: Arc<TurboPuffer>) -> Router {
    Router::new()
        .route("/v1/vectors/:namespace/upsert", post(handle_upsert))
        .route("/v1/vectors/:namespace/query", post(handle_query))
        .route("/v1/vectors/:namespace", get(handle_namespace_info))
        .with_state(turbo)
}

async fn handle_upsert(
    Path(namespace): Path<String>,
    State(turbo): State<Arc<TurboPuffer>>,
    Json(documents): Json<Vec<Document>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match turbo.upsert(&namespace, documents).await {
        Ok(_) => Ok(Json(serde_json::json!({"status": "success"}))),
        Err(e) => {
            warn!("Upsert error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn handle_query(
    Path(namespace): Path<String>,
    State(turbo): State<Arc<TurboPuffer>>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, StatusCode> {
    match turbo.query(&namespace, request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => {
            warn!("Query error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn handle_namespace_info(
    Path(namespace): Path<String>,
    State(turbo): State<Arc<TurboPuffer>>,
) -> Json<serde_json::Value> {
    if let Some(ns) = turbo.namespaces.get(&namespace) {
        Json(serde_json::json!({
            "name": ns.name,
            "document_count": ns.documents.len(),
        }))
    } else {
        Json(serde_json::json!({"error": "namespace not found"}))
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    
    info!("🚀 Starting TurboPuffer (Rust Edition)...");
    
    let turbo = Arc::new(TurboPuffer::new());
    
    info!("📝 Generating and inserting 1,000,000 test documents...");
    
    let start = Instant::now();
    let categories = vec!["tech", "science", "business", "health", "finance", "education"];
    
    let batch_size = 10_000;
    let total_docs = 1_000_000;
    
    for batch_idx in 0..total_docs / batch_size {
        let docs: Vec<Document> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let doc_id = batch_idx * batch_size + i;
                let category_idx = doc_id % 6;
                let vector = generate_realistic_vector(128, category_idx);
                
                Document {
                    id: format!("doc_{}", doc_id),
                    vector: Some(vector),
                    text: Some(format!(
                        "Document {} about {} with advanced content",
                        doc_id, categories[category_idx]
                    )),
                    metadata: Some(serde_json::json!({
                        "category": categories[category_idx],
                        "batch": batch_idx,
                        "priority": doc_id % 10,
                    })),
                }
            })
            .collect();
        
        turbo.upsert("benchmark", docs).await.unwrap();
        
        if (batch_idx + 1) % 10 == 0 {
            let elapsed = start.elapsed();
            let docs_inserted = (batch_idx + 1) * batch_size;
            let rate = docs_inserted as f64 / elapsed.as_secs_f64();
            info!(
                "Progress: {}/{} ({:.1}%) - {:.0} docs/sec",
                docs_inserted,
                total_docs,
                docs_inserted as f64 * 100.0 / total_docs as f64,
                rate
            );
        }
    }
    
    let insert_time = start.elapsed();
    info!(
        "✅ Inserted {} documents in {:?} ({:.0} docs/sec)",
        total_docs,
        insert_time,
        total_docs as f64 / insert_time.as_secs_f64()
    );
    
    info!("🔧 Building vector index...");
    let start_index = Instant::now();
    
    if let Some(ns) = turbo.namespaces.get("benchmark") {
        ns.simple_index.write().build_index(&ns.documents);
    }
    
    info!("✅ Index built in {:?}", start_index.elapsed());
    
    info!("🔥 Running performance benchmarks...");
    
    let test_vector = generate_random_vector(128);
    let _cold_start = Instant::now();
    let cold_response = turbo.query("benchmark", QueryRequest {
        vector: Some(test_vector.clone()),
        text: None,
        filters: None,
        top_k: 10,
        include_text: false,
    }).await.unwrap();
    
    info!("❄️  Cold query latency: {:.2}ms (cache hit: {}, results: {})", 
          cold_response.latency_ms, cold_response.cache_hit, cold_response.results.len());
    
    let warm_response = turbo.query("benchmark", QueryRequest {
        vector: Some(test_vector),
        text: None,
        filters: None,
        top_k: 10,
        include_text: false,
    }).await.unwrap();
    
    info!("🔥 Warm query latency: {:.2}ms (cache hit: {}, results: {})", 
          warm_response.latency_ms, warm_response.cache_hit, warm_response.results.len());
    
    info!("\n🔥 Warm query performance (different vectors):");
    let mut warm_latencies = Vec::new();
    
    for i in 0..10 {
        let query_vector = generate_random_vector(128);
        let response = turbo.query("benchmark", QueryRequest {
            vector: Some(query_vector),
            text: None,
            filters: None,
            top_k: 10,
            include_text: false,
        }).await.unwrap();
        
        warm_latencies.push(response.latency_ms);
        
        if i == 0 {
            info!("First warm query: {:.2}ms", response.latency_ms);
        }
    }
    
    warm_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = warm_latencies[warm_latencies.len() / 2];
    let p95 = warm_latencies[(warm_latencies.len() as f64 * 0.95) as usize];
    let p99 = warm_latencies[(warm_latencies.len() as f64 * 0.99) as usize];
    let avg = warm_latencies.iter().sum::<f64>() / warm_latencies.len() as f64;
    
    info!("\n📊 Warm Query Statistics (Different Vectors):");
    info!("   Min:    {:.2}ms", warm_latencies[0]);
    info!("   P50:    {:.2}ms (target: ~16ms)", p50);
    info!("   Avg:    {:.2}ms", avg);
    info!("   P95:    {:.2}ms", p95);
    info!("   P99:    {:.2}ms", p99);
    info!("   Max:    {:.2}ms", warm_latencies.last().unwrap());
    
    info!("\n🔍 Hybrid search performance:");
    let hybrid_vector = generate_random_vector(128);
    let _hybrid_start = Instant::now();
    let hybrid_response = turbo.query("benchmark", QueryRequest {
        vector: Some(hybrid_vector),
        text: Some("technology innovation".to_string()),
        filters: None,
        top_k: 10,
        include_text: true,
    }).await.unwrap();
    
    info!("Hybrid search latency: {:.2}ms (results: {})",
        hybrid_response.latency_ms, hybrid_response.results.len());
    
    let app = create_app(turbo.clone()).await;
    let addr: std::net::SocketAddr = "0.0.0.0:8080".parse().unwrap();
    
    info!("\n🌐 TurboPuffer HTTP API running on http://{}", addr);
    info!("\n📚 API Examples:");
    info!("Upsert: curl -X POST http://localhost:8080/v1/vectors/demo/upsert -H 'Content-Type: application/json' -d '[{{\"id\":\"1\",\"vector\":[0.1,0.2,0.3,0.4]}}]'");
    info!("Query: curl -X POST http://localhost:8080/v1/vectors/demo/query -H 'Content-Type: application/json' -d '{{\"vector\":[0.1,0.2,0.3,0.4],\"top_k\":10}}'");
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn generate_realistic_vector(dim: usize, cluster: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let cluster_centers = vec![
        vec![0.8, 0.2, -0.1, 0.5, 0.3, -0.2, 0.7, 0.1],
        vec![0.1, 0.9, 0.4, -0.3, 0.6, 0.2, -0.1, 0.8],
        vec![-0.2, 0.3, 0.8, 0.1, -0.4, 0.7, 0.2, 0.5],
        vec![0.6, -0.1, 0.2, 0.9, 0.1, -0.3, 0.5, 0.4],
        vec![0.3, 0.7, -0.2, 0.4, 0.8, 0.1, -0.1, 0.6],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    
    let center = &cluster_centers[cluster % cluster_centers.len()];
    let mut vector = Vec::with_capacity(dim);
    
    for i in 0..dim {
        let base = if i < center.len() { center[i] } else { 0.0 };
        let noise = rng.gen_range(-0.2..0.2);
        vector.push(base + noise);
    }
    
    normalize_vector(&mut vector);
    vector
}

fn generate_random_vector(dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let cluster = rng.gen_range(0..6);
    generate_realistic_vector(dim, cluster)
}

fn normalize_vector(vector: &mut Vec<f32>) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vector.iter_mut() {
            *v /= norm;
        }
    }
}
