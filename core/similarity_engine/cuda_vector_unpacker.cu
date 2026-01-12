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

    // Unpack binary dimensions (1 byte at offset 0)
    uint8_t binary_byte = record[0];
    output[idx*32 + 9] = (binary_byte & 1) ? 1.0f : 0.0f;        // mode
    output[idx*32 + 11] = (binary_byte & 2) ? 1.0f : 0.0f;       // time_sig_4_4
    output[idx*32 + 12] = (binary_byte & 4) ? 1.0f : 0.0f;       // time_sig_3_4
    output[idx*32 + 13] = (binary_byte & 8) ? 1.0f : 0.0f;       // time_sig_5_4
    output[idx*32 + 14] = (binary_byte & 16) ? 1.0f : 0.0f;      // time_sig_other

    // Unpack scaled dimensions (22 uint16 at offset 1)
    const uint16_t* scaled = reinterpret_cast<const uint16_t*>(record + 1);
    // First 9 scaled dimensions (0.0001 precision)
    output[idx*32 + 0] = scaled[0] / 10000.0f;   // acousticness
    output[idx*32 + 1] = scaled[1] / 10000.0f;   // instrumentalness
    output[idx*32 + 2] = scaled[2] / 10000.0f;   // speechiness
    output[idx*32 + 3] = scaled[3] / 10000.0f;   // valence
    output[idx*32 + 4] = scaled[4] / 10000.0f;   // danceability
    output[idx*32 + 5] = scaled[5] / 10000.0f;   // energy
    output[idx*32 + 6] = scaled[6] / 10000.0f;   // liveness
    output[idx*32 + 8] = scaled[7] / 10000.0f;   // key
    output[idx*32 + 16] = scaled[8] / 10000.0f;  // release_date

    // Next 13 scaled dimensions - meta-genre (0.0001 precision)
    output[idx*32 + 19] = scaled[9] / 10000.0f;   // meta-genre1
    output[idx*32 + 20] = scaled[10] / 10000.0f;  // meta-genre2
    output[idx*32 + 21] = scaled[11] / 10000.0f;  // meta-genre3
    output[idx*32 + 22] = scaled[12] / 10000.0f;  // meta-genre4
    output[idx*32 + 23] = scaled[13] / 10000.0f;  // meta-genre5
    output[idx*32 + 24] = scaled[14] / 10000.0f;  // meta-genre6
    output[idx*32 + 25] = scaled[15] / 10000.0f;  // meta-genre7
    output[idx*32 + 26] = scaled[16] / 10000.0f;  // meta-genre8
    output[idx*32 + 27] = scaled[17] / 10000.0f;  // meta-genre9
    output[idx*32 + 28] = scaled[18] / 10000.0f;  // meta-genre10
    output[idx*32 + 29] = scaled[19] / 10000.0f;  // meta-genre11
    output[idx*32 + 30] = scaled[20] / 10000.0f;  // meta-genre12
    output[idx*32 + 31] = scaled[21] / 10000.0f;  // meta-genre13

    // Unpack FP32 dimensions (5 floats at offset 45)
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
