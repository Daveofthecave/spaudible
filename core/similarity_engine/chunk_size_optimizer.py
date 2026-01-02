# core/similarity_engine/chunk_size_optimizer.py
import time
import numpy as np
from .vector_math import VectorOps

class ChunkSizeOptimizer:
    """Dynamically determines optimal chunk size for CPU processing"""
    
    _global_optimal_chunk_size = None
    
    def __init__(self, vector_reader, sample_size=500_000):
        """
        Initialize CPU chunk size optimizer.
        
        Args:
            vector_reader: VectorReader instance
            sample_size: Number of vectors to use for benchmarking
        """
        self.reader = vector_reader
        self.sample_size = min(sample_size, vector_reader.get_total_vectors())
        self.candidates = self._generate_candidates()

        # Use global optimal size if available
        if ChunkSizeOptimizer._global_optimal_chunk_size:
            self.best_size = ChunkSizeOptimizer._global_optimal_chunk_size
            self.optimized = True
        else:
            self.best_size = 100_000  # Default fallback
            self.optimized = False
        
    def _generate_candidates(self):
        """Generate candidate chunk sizes optimized for CPU processing."""
        # CPU-optimized chunk sizes
        return [5_000, 10_000, 15_000, 20_000, 30_000, 50_000, 
                75_000, 100_000, 125_000, 150_000, 200_000, 
                300_000, 500_000]
        
    def optimize(self):
        """Run optimization benchmark for CPU processing."""
        if self.optimized:
            return self.best_size
            
        print("🔧 Calibrating optimal CPU chunk size...")
        query_vector = np.random.rand(32).astype(np.float32)
        vector_ops = VectorOps()
        results = []
        
        for size in self.candidates:
            # Skip sizes larger than total vectors
            if size > self.reader.get_total_vectors():
                continue
                
            # Warmup
            self.run_test(vector_ops, query_vector, size, warmup=True)
            
            # Timed test
            speed = self.run_test(vector_ops, query_vector, size)
            results.append((size, speed))
            print(f"   Chunk {size:12,}: {speed/1e6:.2f}M vec/sec")
        
        # Find fastest candidate
        if results:
            self.best_size = max(results, key=lambda x: x[1])[0]
        else:
            self.best_size = 100_000
            
        self.optimized = True

        # Store in class-level cache for future instances
        ChunkSizeOptimizer._global_optimal_chunk_size = self.best_size    
        
        print(f"   Optimal CPU chunk size: {self.best_size:,} ({(max(results, key=lambda x: x[1])[1]/1e6):.2f}M vec/sec)")
        return self.best_size
        
    def run_test(self, vector_ops, query_vector, chunk_size, warmup=False):
        """Run performance test for a chunk size"""
        start_idx = 0
        processed = 0
        test_size = min(self.sample_size, 500_000) if warmup else self.sample_size
        
        start_time = time.time()
        while processed < test_size:
            read_size = min(chunk_size, test_size - processed)
            vectors = self.reader.read_chunk(start_idx, read_size)
            _ = vector_ops.masked_cosine_similarity_batch(query_vector, vectors)
            processed += read_size
            start_idx = (start_idx + read_size) % self.reader.get_total_vectors()
            
        return processed / (time.time() - start_time)
