# scripts/benchmark_vector_stores.py
"""
Comprehensive benchmark: FAISS vs Pinecone
Measures speed, memory, scalability
"""
from dotenv import load_dotenv
load_dotenv()
import sys
import time
import os
from pathlib import Path
import tracemalloc
from statistics import mean, stdev

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.pdf_processor import PDFProcessor
from app.services.vector_store import VectorStore
from app.services.vector_store_pinecone import PineconeVectorStore


# Test queries
TEST_QUERIES = [
    "What are the HbA1c measurement requirements for diabetes device clinical trials?",
    "What sample size does FDA recommend for early feasibility studies?",
    "What are the safety endpoint requirements for T2DM device trials?",
    "What are the inclusion criteria for diabetes patients?",
    "How should adverse events be monitored?",
]


def build_chunks():
    """Load PDF chunks once."""
    pdf_dir = project_root / "data" / "input" / "protocols"
    proc = PDFProcessor()
    return proc.process_directory_with_chunks(str(pdf_dir))


def benchmark_faiss(chunks):
    """Benchmark FAISS vector store."""
    print("\n" + "="*70)
    print("⚡ BENCHMARKING FAISS (Local)")
    print("="*70)
    
    results = {}
    
    # 1. Index Build Time
    print("\n1️⃣ Index Build Time...")
    tracemalloc.start()
    start = time.time()
    
    vs = VectorStore(
        embedding_model="ncbi/MedCPT-Article-Encoder",
        query_encoder="ncbi/MedCPT-Query-Encoder",
        vector_db_path=project_root / "data" / "vectordb_faiss_benchmark",
    )
    vs.build_index(chunks)
    vs.save()
    
    build_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    results['build_time'] = build_time
    results['memory_mb'] = peak / 1024 / 1024
    
    print(f"   ✅ Build time: {build_time:.2f}s")
    print(f"   💾 Peak memory: {results['memory_mb']:.1f} MB")
    
    # 2. Query Latency (cold start)
    print("\n2️⃣ Query Latency (Cold Start)...")
    start = time.time()
    vs.search(TEST_QUERIES[0], top_k=5)
    cold_latency = time.time() - start
    results['cold_latency_ms'] = cold_latency * 1000
    print(f"   ✅ Cold query: {cold_latency*1000:.1f} ms")
    
    # 3. Query Latency (warm - average of multiple queries)
    print("\n3️⃣ Query Latency (Warm - Average of 5 queries)...")
    latencies = []
    for query in TEST_QUERIES:
        start = time.time()
        vs.search(query, top_k=5)
        latencies.append((time.time() - start) * 1000)
    
    results['avg_latency_ms'] = mean(latencies)
    results['std_latency_ms'] = stdev(latencies) if len(latencies) > 1 else 0
    results['min_latency_ms'] = min(latencies)
    results['max_latency_ms'] = max(latencies)
    
    print(f"   ✅ Average: {results['avg_latency_ms']:.1f} ms")
    print(f"   📊 Min: {results['min_latency_ms']:.1f} ms")
    print(f"   📊 Max: {results['max_latency_ms']:.1f} ms")
    print(f"   📊 Std: {results['std_latency_ms']:.1f} ms")
    
    # 4. Throughput
    print("\n4️⃣ Throughput (queries per second)...")
    num_queries = 20
    start = time.time()
    for i in range(num_queries):
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        vs.search(query, top_k=5)
    elapsed = time.time() - start
    results['qps'] = num_queries / elapsed
    print(f"   ✅ Throughput: {results['qps']:.1f} queries/sec")
    
    return results


def benchmark_pinecone(chunks):
    """Benchmark Pinecone vector store."""
    print("\n" + "="*70)
    print("☁️  BENCHMARKING PINECONE (Cloud)")
    print("="*70)
    
    results = {}
    
    # 1. Index Build Time (includes upload)
    print("\n1️⃣ Index Build Time (includes cloud upload)...")
    start = time.time()
    
    vs = PineconeVectorStore(
        index_name="siro",
        embedding_model="ncbi/MedCPT-Article-Encoder",
        query_encoder="ncbi/MedCPT-Query-Encoder",
    )
    vs.build_index(chunks)
    
    build_time = time.time() - start
    results['build_time'] = build_time
    print(f"   ✅ Build + Upload time: {build_time:.2f}s")
    
    # 2. Query Latency (cold start)
    print("\n2️⃣ Query Latency (Cold Start)...")
    start = time.time()
    vs.search(TEST_QUERIES[0], top_k=5)
    cold_latency = time.time() - start
    results['cold_latency_ms'] = cold_latency * 1000
    print(f"   ✅ Cold query: {cold_latency*1000:.1f} ms")
    
    # 3. Query Latency (warm)
    print("\n3️⃣ Query Latency (Warm - Average of 5 queries)...")
    latencies = []
    for query in TEST_QUERIES:
        start = time.time()
        vs.search(query, top_k=5)
        latencies.append((time.time() - start) * 1000)
    
    results['avg_latency_ms'] = mean(latencies)
    results['std_latency_ms'] = stdev(latencies) if len(latencies) > 1 else 0
    results['min_latency_ms'] = min(latencies)
    results['max_latency_ms'] = max(latencies)
    
    print(f"   ✅ Average: {results['avg_latency_ms']:.1f} ms")
    print(f"   📊 Min: {results['min_latency_ms']:.1f} ms")
    print(f"   📊 Max: {results['max_latency_ms']:.1f} ms")
    print(f"   📊 Std: {results['std_latency_ms']:.1f} ms")
    
    # 4. Throughput
    print("\n4️⃣ Throughput (queries per second)...")
    num_queries = 20
    start = time.time()
    for i in range(num_queries):
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        vs.search(query, top_k=5)
    elapsed = time.time() - start
    results['qps'] = num_queries / elapsed
    print(f"   ✅ Throughput: {results['qps']:.1f} queries/sec")
    
    return results


def print_comparison(faiss_results, pinecone_results):
    """Print side-by-side comparison table."""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<35} {'FAISS':<20} {'Pinecone':<20} {'Winner':<10}")
    print("-"*70)
    
    # Build time
    faiss_build = faiss_results['build_time']
    pine_build = pinecone_results['build_time']
    winner = "FAISS" if faiss_build < pine_build else "Pinecone"
    print(f"{'Index Build Time':<35} {faiss_build:.2f}s{'':<14} {pine_build:.2f}s{'':<14} {winner:<10}")
    
    # Memory (FAISS only)
    print(f"{'Peak Memory Usage':<35} {faiss_results['memory_mb']:.1f} MB{'':<12} N/A (cloud){'':<11} FAISS")
    
    # Cold latency
    faiss_cold = faiss_results['cold_latency_ms']
    pine_cold = pinecone_results['cold_latency_ms']
    winner = "FAISS" if faiss_cold < pine_cold else "Pinecone"
    print(f"{'Cold Query Latency':<35} {faiss_cold:.1f} ms{'':<13} {pine_cold:.1f} ms{'':<13} {winner:<10}")
    
    # Avg latency
    faiss_avg = faiss_results['avg_latency_ms']
    pine_avg = pinecone_results['avg_latency_ms']
    winner = "FAISS" if faiss_avg < pine_avg else "Pinecone"
    print(f"{'Average Query Latency':<35} {faiss_avg:.1f} ms{'':<13} {pine_avg:.1f} ms{'':<13} {winner:<10}")
    
    # Min/Max latency
    print(f"{'Min Query Latency':<35} {faiss_results['min_latency_ms']:.1f} ms{'':<13} {pinecone_results['min_latency_ms']:.1f} ms")
    print(f"{'Max Query Latency':<35} {faiss_results['max_latency_ms']:.1f} ms{'':<13} {pinecone_results['max_latency_ms']:.1f} ms")
    
    # Throughput
    faiss_qps = faiss_results['qps']
    pine_qps = pinecone_results['qps']
    winner = "FAISS" if faiss_qps > pine_qps else "Pinecone"
    print(f"{'Throughput (queries/sec)':<35} {faiss_qps:.1f}{'':<16} {pine_qps:.1f}{'':<16} {winner:<10}")
    
    print("-"*70)
    
    # Summary
    print("\n🏆 WINNER ANALYSIS:")
    print("-"*70)
    
    faiss_wins = 0
    pine_wins = 0
    
    if faiss_build < pine_build:
        faiss_wins += 1
        print("✅ FAISS: Faster index building")
    else:
        pine_wins += 1
        print("✅ Pinecone: Faster index building")
    
    if faiss_cold < pine_cold:
        faiss_wins += 1
        print("✅ FAISS: Lower cold-start latency")
    else:
        pine_wins += 1
        print("✅ Pinecone: Lower cold-start latency")
    
    if faiss_avg < pine_avg:
        faiss_wins += 1
        print("✅ FAISS: Lower average latency")
    else:
        pine_wins += 1
        print("✅ Pinecone: Lower average latency")
    
    if faiss_qps > pine_qps:
        faiss_wins += 1
        print("✅ FAISS: Higher throughput")
    else:
        pine_wins += 1
        print("✅ Pinecone: Higher throughput")
    
    print()
    if faiss_wins > pine_wins:
        print(f"🏆 OVERALL WINNER: FAISS ({faiss_wins}/{faiss_wins + pine_wins} metrics)")
    elif pine_wins > faiss_wins:
        print(f"🏆 OVERALL WINNER: Pinecone ({pine_wins}/{faiss_wins + pine_wins} metrics)")
    else:
        print("🤝 TIE: Both perform similarly!")


def main():
    print("="*70)
    print("🔬 COMPREHENSIVE VECTOR STORE BENCHMARK")
    print("="*70)
    print("Testing: FAISS (local) vs Pinecone (cloud)")
    print("Dataset: FDA Guidelines (53 chunks)")
    print("Embeddings: MedCPT (768-dim)")
    print()
    
    # Load data
    chunks = build_chunks()
    
    # Benchmark both
    faiss_results = benchmark_faiss(chunks)
    pinecone_results = benchmark_pinecone(chunks)
    
    # Compare
    print_comparison(faiss_results, pinecone_results)
    
    # Additional considerations
    print("\n💡 ADDITIONAL CONSIDERATIONS:")
    print("-"*70)
    print("📍 FAISS Advantages:")
    print("   • Zero cost (free)")
    print("   • No network latency")
    print("   • Works offline")
    print("   • Full control over data")
    print()
    print("☁️  Pinecone Advantages:")
    print("   • Scales to billions of vectors")
    print("   • Managed service (no maintenance)")
    print("   • Multi-user/distributed access")
    print("   • Automatic backups")
    print("   • High availability (99.9% uptime)")
    print()
    print("💰 COST COMPARISON:")
    print("   • FAISS: $0 (free)")
    print("   • Pinecone: ~$70/month for 1M vectors (starter plan)")
    print()
    print("📈 SCALABILITY:")
    print("   • FAISS: Limited by RAM (~10M vectors on 32GB)")
    print("   • Pinecone: Unlimited (billions of vectors)")
    print("="*70)


if __name__ == "__main__":
    main()
