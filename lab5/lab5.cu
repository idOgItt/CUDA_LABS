#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

constexpr uint32_t BLOCK_SIZE = 256;
constexpr uint32_t GRID_SIZE  = 1024;

__global__ void predicate_kernel(
    const uint32_t* in,
    uint32_t* pred,
    uint32_t bit,
    uint32_t N)
{
    uint32_t id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    uint32_t stride = BLOCK_SIZE * GRID_SIZE;
    while (id < N) {
        uint32_t v = (in[id] >> bit) & 1;
        pred[id] = v ? 0 : 1;
        id += stride;
    }
}

__global__ void scatter_kernel(
    const uint32_t* in,
    const uint32_t* pred,
    const uint32_t* scan,
    uint32_t* out,
    uint32_t totalZeros,
    uint32_t N)
{
    uint32_t id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    uint32_t stride = BLOCK_SIZE * GRID_SIZE;
    while (id < N) {
        if (pred[id]) {
            out[scan[id]] = in[id];
        } else {
            out[totalZeros + (id - scan[id])] = in[id];
        }
        id += stride;
    }
}

int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    uint32_t N;
    if (!std::cin.read(reinterpret_cast<char*>(&N), sizeof(N)))
        return 0;
    std::vector<uint32_t> h(N);
    std::cin.read(reinterpret_cast<char*>(h.data()), N * sizeof(uint32_t));

    uint32_t *d_in, *d_out, *d_pred, *d_scan;
    cudaMalloc(&d_in,   N * sizeof(uint32_t));
    cudaMalloc(&d_out,  N * sizeof(uint32_t));
    cudaMalloc(&d_pred, N * sizeof(uint32_t));
    cudaMalloc(&d_scan, N * sizeof(uint32_t));
    cudaMemcpy(d_in, h.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    thrust::device_ptr<uint32_t> t_pred(d_pred);
    thrust::device_ptr<uint32_t> t_scan(d_scan);

    for (uint32_t bit = 0; bit < 32; ++bit) {
        predicate_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_pred, bit, N);
        cudaDeviceSynchronize();

        thrust::exclusive_scan(t_pred, t_pred + N, t_scan);

        uint32_t last_pred, last_scan;
        cudaMemcpy(&last_pred, d_pred + (N - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_scan, d_scan + (N - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t totalZeros = last_pred + last_scan;

        scatter_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
            d_in, d_pred, d_scan, d_out, totalZeros, N
        );
        cudaDeviceSynchronize();

        std::swap(d_in, d_out);
    }

    cudaMemcpy(h.data(), d_in, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout.write(reinterpret_cast<char*>(h.data()), N * sizeof(uint32_t));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_pred);
    cudaFree(d_scan);
    return 0;
}
