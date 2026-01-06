# core/similarity_engine/chunk_size_optimizer.py
import time
import numpy as np
from .vector_math import VectorOps

class ChunkSizeOptimizer:
    """Improved optimizer with realistic workload simulation"""
    
    _global_optimal_chunk_size = None
    
    def __init__(self, vector_reader, sample_size=1_000_000):
        self.reader = vector_reader
        
        # Get total vectors using either method or attribute
        if hasattr(vector_reader, 'get_total_vectors'):
            total_vectors = vector_reader.get_total_vectors()
        elif hasattr(vector_reader, 'total_vectors'):
            total_vectors = vector_reader.total_vectors
        else:
            raise AttributeError("VectorReader has no total_vectors attribute or method")
            
        self.sample_size = min(sample_size, total_vectors)
        self.candidates = self._generate_candidates()
        self.best_size = 100_000
        self.optimized = False

    def _generate_candidates(self):
        """Generate candidate chunk sizes optimized for CPU processing."""
        # CPU-optimized chunk sizes
        return [1_000_000, 750_000, 500_000, 300_000, 200_000, 150_000, 
                125_000, 100_000, 75_000, 50_000, 30_000, 20_000, 
                15_000, 10_000, 5_000]
        
    def optimize(self):
        """Run optimization with realistic workload"""
        if self.optimized:
            return self.best_size
            
        print("🔧 Calibrating optimal CPU chunk size...")
        query_vector = np.random.rand(32).astype(np.float32)
        vector_ops = VectorOps()
        results = []
        
        # Use distributed sample points across the dataset
        sample_points = np.linspace(
            0, 
            self.reader.get_total_vectors() - max(self.candidates), 
            10,
            dtype=int
        )
        
        for size in self.candidates:
            total_speed = 0
            valid_tests = 0
            
            for start_idx in sample_points:
                try:
                    # Warmup
                    self._run_test(vector_ops, query_vector, start_idx, size, warmup=True)
                    
                    # Timed test
                    speed = self._run_test(vector_ops, query_vector, start_idx, size)
                    total_speed += speed
                    valid_tests += 1
                except Exception:
                    continue
            
            if valid_tests > 0:
                avg_speed = total_speed / valid_tests
                results.append((size, avg_speed))
                print(f"   Chunk size: {size:10,}: {avg_speed/1e6:.2f}M vec/sec")
        
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
        
    def _run_test(self, vector_ops, query_vector, start_idx, chunk_size, warmup=False):
        """Run performance test at different file positions"""
        test_size = min(chunk_size, 100_000) if warmup else chunk_size
        
        # Read vectors and masks
        vectors = self.reader.read_chunk(start_idx, test_size)
        masks = self.reader.read_masks(start_idx, test_size)
        
        # Compute similarity
        start_time = time.time()
        _ = vector_ops.compute_similarity(query_vector, vectors, masks)
        elapsed = time.time() - start_time
        
        return test_size / elapsed
