# core/utilities/gpu_utils.py
import torch
from config import VRAM_SAFETY_FACTOR

def get_gpu_info():
    """Get available GPU information including VRAM."""
    if not torch.cuda.is_available():
        return None
    
    device_count = torch.cuda.device_count()
    gpu_info = []
    
    for i in range(device_count):
        device = torch.device(f'cuda:{i}')
        props = torch.cuda.get_device_properties(device)
        free_vram = torch.cuda.mem_get_info(device)[0]
        total_vram = props.total_memory
        
        gpu_info.append({
            "id": i,
            "name": props.name,
            "total_vram": total_vram,
            "free_vram": free_vram,
            "capability": props.major,
            "multi_processor_count": props.multi_processor_count
        })
    
    return gpu_info

def recommend_max_batch_size(vector_dim=32, dtype_size=4, safety_factor=VRAM_SAFETY_FACTOR):
    """Recommend max batch size based on available VRAM."""
    gpu_info = get_gpu_info()
    if not gpu_info:
        return 0  # No GPU available
    
    # Calculate VRAM per vector (bytes)
    bytes_per_vector = vector_dim * dtype_size
    
    # Use first GPU
    free_vram = gpu_info[0]['free_vram']
    usable_vram = free_vram * safety_factor
    
    # Calculate max batch size
    max_batch = int(usable_vram // bytes_per_vector)
    return max_batch

def print_gpu_info():
    """Print GPU information for diagnostics."""
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("  No GPU devices available")
        return
    
    print("  Detected GPU devices:")
    for i, gpu in enumerate(gpu_info):
        print(f"    GPU {i}: {gpu['name']}")
        print(f"      VRAM: {gpu['free_vram']/(1024**3):.1f} GB free of {gpu['total_vram']/(1024**3):.1f} GB")
        print(f"      Compute Capability: {gpu['capability']}.x")
