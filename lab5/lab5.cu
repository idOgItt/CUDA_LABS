#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr uint32_t BLOCK_SIZE = 256;

__global__ void predicate_kernel(const uint32_t *in, uint32_t *pred, uint32_t bit, uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        uint32_t v = (in[i] >> bit) & 1;
        pred[i] = v ? 0 : 1;
    }
}

__global__ void scatter_kernel(const uint32_t *in, const uint32_t *pred, const uint32_t *scan, uint32_t *out,
                               uint32_t totalZeros, uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (pred[i]) {
            out[scan[i]] = in[i];
        } else {
            out[totalZeros + (i - scan[i])] = in[i];
        }
    }
}

__device__ __forceinline__ uint32_t cbc(uint32_t i) {
    return i + (i >> 5);
}

__global__ void scan_block(const uint32_t *in, uint32_t *out, uint32_t *sums, uint32_t N) {
    extern __shared__ uint32_t s[];
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x;
    uint32_t start = gid * blockDim.x * 2 + tid;
    uint32_t lr = blockDim.x;
    uint32_t ai = start < N ? in[start] : 0;
    uint32_t bi = start + lr < N ? in[start + lr] : 0;
    s[cbc(tid)] = ai;
    s[cbc(tid + lr)] = bi;
    for (uint32_t offset = 1, d = lr; d > 0; d >>= 1, offset <<= 1) {
        __syncthreads();
        if (tid < d) {
            uint32_t idx1 = cbc(offset * (2 * tid + 1) - 1);
            uint32_t idx2 = cbc(offset * (2 * tid + 2) - 1);
            s[idx2] += s[idx1];
        }
    }
    if (tid == 0) {
        uint32_t idx = cbc(2 * lr - 1);
        sums[gid] = s[idx];
        s[idx] = 0;
    }
    for (uint32_t offset = lr, d = 1; d < 2 * lr; d <<= 1, offset >>= 1) {
        __syncthreads();
        if (tid < d) {
            uint32_t idx1 = cbc(offset * (2 * tid + 1) - 1);
            uint32_t idx2 = cbc(offset * (2 * tid + 2) - 1);
            uint32_t t = s[idx1];
            s[idx1] = s[idx2];
            s[idx2] += t;
        }
    }
    __syncthreads();
    if (start < N) out[start] = s[cbc(tid)];
    if (start + lr < N) out[start + lr] = s[cbc(tid + lr)];
}

__global__ void add_sums(uint32_t *data, const uint32_t *sums, uint32_t N) {
    uint32_t gid = blockIdx.x + 1;
    uint32_t start = gid * blockDim.x * 2 + threadIdx.x;
    if (start < N) data[start] += sums[gid];
    if (start + blockDim.x < N) data[start + blockDim.x] += sums[gid];
}

void exclusive_scan(uint32_t *d_in, uint32_t *d_out, uint32_t N) {
    uint32_t t = BLOCK_SIZE;
    uint32_t elems = t * 2;
    uint32_t blocks = (N + elems - 1) / elems;
    uint32_t *d_sums;
    cudaMalloc(&d_sums, blocks * sizeof(uint32_t));
    size_t shm = (elems + elems / 32) * sizeof(uint32_t);
    scan_block<<<blocks, t, shm>>>(d_in, d_out, d_sums, N);
    if (blocks > 1) {
        uint32_t *d_sums_scan;
        cudaMalloc(&d_sums_scan, blocks * sizeof(uint32_t));
        exclusive_scan(d_sums, d_sums_scan, blocks);
        add_sums<<<blocks - 1, t>>>(d_out, d_sums_scan, N);
        cudaFree(d_sums_scan);
    }
    cudaFree(d_sums);
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    uint32_t N;
    std::cin.read(reinterpret_cast<char *>(&N), sizeof(N));
    std::vector<uint32_t> h(N);
    std::cin.read(reinterpret_cast<char *>(h.data()), N * sizeof(uint32_t));

    uint32_t *d_in, *d_out, *d_pred, *d_scan;
    cudaMalloc(&d_in, N * sizeof(uint32_t));
    cudaMalloc(&d_out, N * sizeof(uint32_t));
    cudaMalloc(&d_pred, N * sizeof(uint32_t));
    cudaMalloc(&d_scan, N * sizeof(uint32_t));
    cudaMemcpy(d_in, h.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (uint32_t bit = 0; bit < 32; ++bit) {
        predicate_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_pred, bit, N);
        exclusive_scan(d_pred, d_scan, N);
        uint32_t last_pred, last_scan;
        cudaMemcpy(&last_pred, d_pred + N - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_scan, d_scan + N - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t totalZeros = last_pred + last_scan;
        scatter_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_pred, d_scan, d_out, totalZeros, N);
        std::swap(d_in, d_out);
    }

    cudaMemcpy(h.data(), d_in, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout.write(reinterpret_cast<char *>(h.data()), N * sizeof(uint32_t));

    return 0;
}
