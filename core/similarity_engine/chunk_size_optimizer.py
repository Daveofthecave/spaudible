# core/similarity_engine/chunk_size_optimizer.py
import time
import numpy as np
import torch
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
        
        # CPU-optimized chunk size candidates
        self.candidates = [1_000_000, 750_000, 500_000, 300_000, 200_000, 150_000, 
                          125_000, 100_000, 75_000, 50_000, 30_000, 20_000, 
                          15_000, 10_000, 5_000]
        
        self.best_size = 100_000
        self.optimized = False

    def optimize(self) -> int:
        """Run optimization with realistic workload"""
        if self.optimized:
            return self.best_size
            
        print("🔧 Calibrating optimal CPU chunk size...")
        
        # Use a realistic query vector (not random) for better cache behavior
        query_vector = np.array([
            0.5, 0.3, 0.1, 0.7, 0.8, 0.9, 0.2,  # acousticness through liveness
            -10.0, 0.5, 1.0, 120.0,             # loudness, key, mode, tempo
            1.0, 0.0, 0.0, 0.0,                 # time signatures (4/4, 3/4, 5/4, other)
            3.5, 2000.5, 0.6, 7.5, 1000000.0,   # duration, release year, popularity, followers, genre1
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # genres 2-8
            0.8, 0.0, 0.0, 0.0, 0.0, 0.0        # genres 9-13
        ], dtype=np.float32)
        
        vector_ops = VectorOps()
        results = []
        total_candidates = len(self.candidates)
        
        # Use distributed sample points across the dataset
        max_start = max(0, self.reader.get_total_vectors() - max(self.candidates))
        sample_points = np.linspace(0, max_start, min(3, max(1, self.reader.get_total_vectors() // self.sample_size)), dtype=int)
        
        for idx, size in enumerate(self.candidates):
            total_speed = 0.0
            valid_tests = 0
            
            for start_idx in sample_points:
                try:
                    # Warmup run
                    self._run_test(vector_ops, query_vector, start_idx, size, warmup=True)
                    
                    # Timed test
                    speed = self._run_test(vector_ops, query_vector, start_idx, size)
                    total_speed += speed
                    valid_tests += 1
                    
                except Exception as e:
                    # Log individual test failures but continue
                    print(f"   ⚠️  Chunk size {size:,} test failed at pos {start_idx}: {e}")
                    continue
            
            if valid_tests > 0:
                avg_speed = total_speed / valid_tests
                results.append((size, avg_speed))
                print(f"   Chunk size: {size:10,}: {avg_speed/1e6:.2f}M vec/sec")
        
        # Determine best chunk size
        if results:
            # Found at least one working chunk size
            best_result = max(results, key=lambda x: x[1])
            self.best_size = best_result[0]
            best_speed = best_result[1]
            print(f"   Optimal CPU chunk size: {self.best_size:,} ({best_speed/1e6:.2f}M vec/sec)")
        else:
            # All benchmarks failed - use conservative default
            print(f"   ⚠️  All chunk size benchmarks failed!")
            print(f"      This may indicate a corrupted vector file or I/O error.")
            print(f"      Using safe default chunk size: {self.best_size:,}")
            self.best_size = 100_000
            
        self.optimized = True

        # Cache for future instances
        ChunkSizeOptimizer._global_optimal_chunk_size = self.best_size    
        return self.best_size

    def _run_test(self, vector_ops, query_vector, start_idx, chunk_size, warmup=False):
        """Run performance test at different file positions"""
        test_size = min(chunk_size, 100_000) if warmup else chunk_size
        
        # Read vectors and masks using unified reader
        vectors = self.reader.read_chunk(start_idx, test_size)
        masks = self.reader.read_masks(start_idx, test_size)
        
        if torch.is_tensor(vectors):
            # Move CUDA tensors to CPU first, then convert to numpy
            if vectors.is_cuda:
                vectors = vectors.cpu()
            vectors = vectors.numpy()
        if torch.is_tensor(masks):
            if masks.is_cuda:
                masks = masks.cpu()
            masks = masks.numpy()
        
        # Compute similarity
        start_time = time.time()
        _ = vector_ops.compute_similarity(query_vector, vectors, masks)
        elapsed = time.time() - start_time
        
        return test_size / elapsed
