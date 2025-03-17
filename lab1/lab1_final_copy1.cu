#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

constexpr int32_t kThreadsPerBlock = 1024;

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
__device__ T cuda_min(T a, T b) {
    return min(a, b);
}

template <typename T, typename std::enable_if<std::is_same<T, float>::value, int>::type = 0>
__device__ T cuda_min(T a, T b) {
    return fminf(a, b);
}

template <typename T, typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
__device__ T cuda_min(T a, T b) {
    return fmin(a, b);
}

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
__global__ void VectorPerElemMinKernel(const T* first_vector,
                                       const T* second_vector,
                                       T* result,
                                       int32_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (; idx < size; idx += stride) {
        result[idx] = cuda_min(first_vector[idx], second_vector[idx]);
    }
}

class CudaVectorMin {
public:
    explicit CudaVectorMin(int32_t size) : vector_size_(size) {
        cudaMalloc(&device_vector_a_, vector_size_ * sizeof(double));
        cudaMalloc(&device_vector_b_, vector_size_ * sizeof(double));
        cudaMalloc(&device_result_, vector_size_ * sizeof(double));
    }

    ~CudaVectorMin() {
        cudaFree(device_vector_a_);
        cudaFree(device_vector_b_);
        cudaFree(device_result_);
    }

    void CopyDataToDevice(const std::vector<double>& host_vector_a,
                          const std::vector<double>& host_vector_b) {
        cudaMemcpy(device_vector_a_, host_vector_a.data(), vector_size_ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(device_vector_b_, host_vector_b.data(), vector_size_ * sizeof(double), cudaMemcpyHostToDevice);
    }

    void Compute(int blocks, int threads) {
        VectorPerElemMinKernel<<<blocks, threads>>>(device_vector_a_, device_vector_b_, device_result_, vector_size_);
        cudaDeviceSynchronize();
    }

    void CopyDataToHost(std::vector<double>& host_result) {
        cudaMemcpy(host_result.data(), device_result_, vector_size_ * sizeof(double), cudaMemcpyDeviceToHost);
    }

private:
    int32_t vector_size_;
    double* device_vector_a_;
    double* device_vector_b_;
    double* device_result_;
};

double MeasureCUDA(CudaVectorMin& cuda_vector_min,
                   const std::vector<double>& a,
                   const std::vector<double>& b,
                   int blocks, int threads) {
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cuda_vector_min.CopyDataToDevice(a, b);
    cuda_vector_min.Compute(blocks, threads);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}

double MeasureCPU(const std::vector<double>& a, const std::vector<double>& b) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::min(a[i], b[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void RunTests() {
    std::vector<int> sizes = {1'000, 10'000, 100'000, 1'000'000, 3'000'000};
    std::vector<std::pair<int, int>> gridConfigs = {
        {1, 32}, {32, 32}, {256, 256}, {1024, 1024}
    };

    for (int size : sizes) {
        std::vector<double> vecA(size, 1.0);
        std::vector<double> vecB(size, 2.0);

        double cpuTime = MeasureCPU(vecA, vecB);
        std::cout << "CPU Time (" << size << " elements): " << cpuTime << " ms\n";

        for (auto config : gridConfigs) {
            int blocks = config.first;
            int threads = config.second;

            CudaVectorMin cuda_vector_min(size);
            double cudaTime = MeasureCUDA(cuda_vector_min, vecA, vecB, blocks, threads);

            std::cout << "CUDA Time (" << size << " elements, "
                      << "<<<" << blocks << ", " << threads << ">>>): "
                      << cudaTime << " ms\n";
        }

        std::cout << "--------------------------------\n";
    }
}

int main() {
    try {
        RunTests();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}