#include <iostream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

constexpr int BLOCK_SIZE = 256;
constexpr int SHARED_SIZE = 2 * BLOCK_SIZE;
constexpr int LOG_NUM_BANKS = 5;
constexpr int NUM_BANKS = 1 << LOG_NUM_BANKS;
constexpr int CONFLICT_FREE_PADDING = SHARED_SIZE / NUM_BANKS;
constexpr int PADDED_SIZE = SHARED_SIZE + CONFLICT_FREE_PADDING;
constexpr int GRID_SIZE = 1024;
constexpr int HIST_BINS = 256;

#define PAD_INDEX(idx) ((idx) + ((idx) >> LOG_NUM_BANKS))

__global__ void reduce_kernel(const uint32_t* in, uint32_t* out, uint32_t N) {
    extern __shared__ uint32_t sdata[];
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x * 2 + tid;
    uint32_t sum = 0;
    if (gid < N) sum = in[gid];
    if (gid + blockDim.x < N) sum += in[gid + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void histogram_kernel(const uint32_t* in, uint32_t* histo, uint32_t N) {
    __shared__ uint32_t local_hist[HIST_BINS];
    uint32_t tid = threadIdx.x;
    for (int i = tid; i < HIST_BINS; i += blockDim.x) local_hist[i] = 0;
    __syncthreads();
    uint32_t gid = blockIdx.x * blockDim.x + tid;
    uint32_t stride = gridDim.x * blockDim.x;
    while (gid < N) {
        atomicAdd(&local_hist[in[gid] & (HIST_BINS - 1)], 1);
        gid += stride;
    }
    __syncthreads();
    for (int i = tid; i < HIST_BINS; i += blockDim.x) {
        atomicAdd(&histo[i], local_hist[i]);
    }
}

__global__ void predicate_kernel(const uint32_t* in, uint32_t* pred, uint32_t bit, uint32_t N) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    while (idx < N) {
        pred[idx] = (((in[idx] >> bit) & 1u) == 0u);
        idx += stride;
    }
}

__global__ void scatter_kernel(const uint32_t* in, const uint32_t* pred, const uint32_t* scan, uint32_t* out, uint32_t totalZeros, uint32_t N) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    while (idx < N) {
        if (pred[idx]) out[scan[idx]] = in[idx];
        else out[totalZeros + idx - scan[idx]] = in[idx];
        idx += stride;
    }
}

__global__ void blelloch_scan_block(uint32_t* data, uint32_t* blockSums, int N) {
    extern __shared__ uint32_t temp[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int start = 2 * bid * BLOCK_SIZE;
    int ai = tid;
    int bi = tid + BLOCK_SIZE;
    int ai_pad = PAD_INDEX(ai);
    int bi_pad = PAD_INDEX(bi);
    temp[ai_pad] = (start + ai < N) ? data[start + ai] : 0u;
    temp[bi_pad] = (start + bi < N) ? data[start + bi] : 0u;
    __syncthreads();
    for (int d = 0; d < LOG_NUM_BANKS + 4; ++d) {
        int offset = 1 << (d + 1);
        int idx = (tid + 1) * offset - 1;
        int idx_pad = PAD_INDEX(idx);
        int left = idx - (offset >> 1);
        int left_pad = PAD_INDEX(left);
        if (idx < SHARED_SIZE) temp[idx_pad] += temp[left_pad];
        __syncthreads();
    }
    if (tid == 0) {
        int last = PAD_INDEX(SHARED_SIZE - 1);
        blockSums[bid] = temp[last];
        temp[last] = 0u;
    }
    __syncthreads();
    for (int d = LOG_NUM_BANKS + 4; d >= 0; --d) {
        int offset = 1 << (d + 1);
        int idx = (tid + 1) * offset - 1;
        int idx_pad = PAD_INDEX(idx);
        int left = idx - (offset >> 1);
        int left_pad = PAD_INDEX(left);
        if (idx < SHARED_SIZE) {
            uint32_t t = temp[left_pad];
            temp[left_pad] = temp[idx_pad];
            temp[idx_pad] += t;
        }
        __syncthreads();
    }
    if (start + ai < N) data[start + ai] = temp[ai_pad];
    if (start + bi < N) data[start + bi] = temp[bi_pad];
}

__global__ void add_block_offsets(uint32_t* data, const uint32_t* blockSumsScan, int N) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int start = 2 * bid * BLOCK_SIZE;
    uint32_t offset = blockSumsScan[bid];
    if (start + tid < N) data[start + tid] += offset;
    if (start + BLOCK_SIZE + tid < N) data[start + BLOCK_SIZE + tid] += offset;
}

void blelloch_scan(uint32_t* d_data, int N) {
    int numBlocks = (N + SHARED_SIZE - 1) / SHARED_SIZE;
    uint32_t* d_blockSums;
    cudaMalloc(&d_blockSums, numBlocks * sizeof(uint32_t));
    blelloch_scan_block<<<numBlocks, BLOCK_SIZE, PADDED_SIZE * sizeof(uint32_t)>>>(d_data, d_blockSums, N);
    cudaDeviceSynchronize();
    if (numBlocks > 1) {
        blelloch_scan(d_blockSums, numBlocks);
        add_block_offsets<<<numBlocks, BLOCK_SIZE>>>(d_data, d_blockSums, N);
        cudaDeviceSynchronize();
    }
    cudaFree(d_blockSums);
}

int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    uint32_t N;
    if (!std::cin.read(reinterpret_cast<char*>(&N), sizeof(N))) return 0;
    std::vector<uint32_t> h_in(N);
    std::cin.read(reinterpret_cast<char*>(h_in.data()), N * sizeof(uint32_t));
    uint32_t *d_in, *d_out, *d_pred, *d_scan, *d_reduceBuf, *d_histBuf;
    cudaMalloc(&d_in, N * sizeof(uint32_t));
    cudaMalloc(&d_out, N * sizeof(uint32_t));
    cudaMalloc(&d_pred, N * sizeof(uint32_t));
    cudaMalloc(&d_scan, N * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    int reduceBlocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    cudaMalloc(&d_reduceBuf, reduceBlocks * sizeof(uint32_t));
    int histBins = HIST_BINS;
    cudaMalloc(&d_histBuf, histBins * sizeof(uint32_t));
    cudaMemset(d_histBuf, 0, histBins * sizeof(uint32_t));
    for (uint32_t bit = 0; bit < 32; ++bit) {
        predicate_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_pred, bit, N);
        cudaDeviceSynchronize();
        cudaMemcpy(d_scan, d_pred, N * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        blelloch_scan(d_scan, N);
        uint32_t last_pred, last_scan;
        cudaMemcpy(&last_pred, d_pred + (N - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_scan, d_scan + (N - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t totalZeros = last_pred + last_scan;
        scatter_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_pred, d_scan, d_out, totalZeros, N);
        cudaDeviceSynchronize();
        std::swap(d_in, d_out);
    }
    cudaMemcpy(h_in.data(), d_in, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout.write(reinterpret_cast<char*>(h_in.data()), N * sizeof(uint32_t));
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_pred);
    cudaFree(d_scan);
    cudaFree(d_reduceBuf);
    cudaFree(d_histBuf);
    return 0;
}
