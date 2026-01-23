// core/similarity_engine/cuda_vector_unpacker.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void unpack_vectors_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    size_t num_vectors
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vectors) return;

    // Each record is 104 bytes
    const unsigned char* record = input + idx * 104;

    // Initialize all dimensions to -1.0 (sentinel)
    for (int i = 0; i < 32; i++) {
        output[idx*32 + i] = -1.0f;
    }

    // Unpack binary dimensions (1 byte at offset 0)
    uint8_t binary_byte = record[0];
    output[idx*32 + 9] = (binary_byte & 1) ? 1.0f : -1.0f;        // mode
    output[idx*32 + 11] = (binary_byte & 2) ? 1.0f : -1.0f;       // time_sig_4_4
    output[idx*32 + 12] = (binary_byte & 4) ? 1.0f : -1.0f;       // time_sig_3_4
    output[idx*32 + 13] = (binary_byte & 8) ? 1.0f : -1.0f;      // time_sig_5_4
    output[idx*32 + 14] = (binary_byte & 16) ? 1.0f : -1.0f;      // time_sig_other

    // Unpack scaled dimensions (22 uint16 at offset 1-44)
    const uint16_t* scaled = reinterpret_cast<const uint16_t*>(record + 1);
    
    // Map to correct dimensions
    int dim_map[22] = {0,1,2,3,4,5,6,8,16,19,20,21,22,23,24,25,26,27,28,29,30,31};
    
    for (int i = 0; i < 22; i++) {
        float val = scaled[i] / 10000.0f;
        // Convert 0 back to -1.0 sentinel
        if (scaled[i] == 0) val = -1.0f;
        output[idx*32 + dim_map[i]] = val;
    }

    // Unpack FP32 dimensions (5 floats at offset 45-64)
    const float* fp32 = reinterpret_cast<const float*>(record + 45);
    output[idx*32 + 7] = fp32[0];   // loudness
    output[idx*32 + 10] = fp32[1];  // tempo
    output[idx*32 + 15] = fp32[2];  // duration
    output[idx*32 + 17] = fp32[3];  // popularity
    output[idx*32 + 18] = fp32[4];  // followers
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unpack_vectors_kernel", &unpack_vectors_kernel, "Unpack condensed vectors");
}
