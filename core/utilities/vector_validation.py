# core/utilities/vector_validation.py
import math
from config import PathConfig

def validate_vector(vector: List[float]) -> Tuple[bool, str]:
    """Comprehensive vector validation with range checking."""
    if len(vector) != 32:
        return False, f"Vector length {len(vector)} != 32 (in {PathConfig.get_vector_file()})"
    
    for i, val in enumerate(vector):
        # Type check
        if not isinstance(val, (int, float)):
            return False, f"vector[{i}] = {val} (type: {type(val).__name__})"
        
        # NaN/Inf check
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return False, f"vector[{i}] = {val} (NaN or Inf)"
        
        # Range check (-1 is allowed as sentinel)
        if not (-1.0 <= val <= 1.0):
            return False, f"vector[{i}] = {val} (out of [-1,1] range)"
    
    return True, "Valid"
