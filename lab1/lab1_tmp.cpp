#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

template <typename T>
struct CudaDeleter {
    void operator()(T* pointer) const noexcept {
        if (pointer) {
            cudaFree(pointer);
        }
    }
};

template <typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(const std::size_t kSize) {
        T* temp_ptr = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&temp_ptr),
                       kSize * sizeof(T)) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA memory");
        }
        buffer_.reset(temp_ptr);
    }

    [[nodiscard]] T* Get() const noexcept { return buffer_.get(); }

private:
    std::unique_ptr<T, CudaDeleter<T>> buffer_;
};

template <typename T>
__global__ void VectorPerElemMin(const T* first_vector, const T* second_vector,
                                 T* result, int32_t data_size) {
    int32_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < data_size) {
        result[idx] = fmin(first_vector[idx], second_vector[idx]);
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class CudaVectorMin {
public:
    explicit CudaVectorMin(int32_t vector_size)
        : vector_size_(vector_size),
          device_vector_a_(vector_size),
          device_vector_b_(vector_size),
          device_result_(vector_size) {}

    [[nodiscard]] std::vector<T> Compute(const std::vector<T>& host_vector_a,
                                         const std::vector<T>& host_vector_b) {
        if (host_vector_a.size() != vector_size_ ||
            host_vector_b.size() != vector_size_) {
            throw std::invalid_argument("Vector sizes do not match");
        }

        if (cudaMemcpy(device_vector_a_.Get(), host_vector_a.data(),
                       vector_size_ * sizeof(T),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(device_vector_b_.Get(), host_vector_b.data(),
                       vector_size_ * sizeof(T),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error(
                "cudaMemcpy failed while copying data to device");
        }

        constexpr int32_t kBlocksize = 1024;
        const int32_t kNumBlocks = (vector_size_ + kBlocksize - 1) / kBlocksize;

        VectorPerElemMin<<<kNumBlocks, kBlocksize>>>(
            device_vector_a_.Get(), device_vector_b_.Get(),
            device_result_.Get(), vector_size_);

        std::vector<T> host_result(vector_size_);
        if (cudaMemcpy(host_result.data(), device_result_.Get(),
                       vector_size_ * sizeof(T),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error(
                "cudaMemcpy failed while copying result back to host");
        }

        return host_result;
    }

private:
    int32_t vector_size_;
    CudaBuffer<T> device_vector_a_;
    CudaBuffer<T> device_vector_b_;
    CudaBuffer<T> device_result_;
};

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
std::vector<T> ReadVector(const int32_t kSize) {
    std::vector<T> vector(kSize);
    for (T& value : vector) {
        std::cin >> value;
    }
    return vector;
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
void PrintVector(const std::vector<T>& vec) {
    constexpr int32_t kPrecision = 10;
    std::cout.precision(kPrecision);
    for (const T& value : vec) {
        std::cout << std::scientific << value << " ";
    }
    std::cout << std::endl;
}

int32_t ReadInputSize() {
    int32_t input_size;
    std::cin >> input_size;

    if (constexpr int32_t kMaxSize = (1 << 25);
        input_size <= 0 || input_size >= kMaxSize) {
        throw std::invalid_argument("Incorrect size of input data");
    }

    return input_size;
}

int main() {
    try {
        const int32_t kInputSize = ReadInputSize();

        const std::vector<double> kHostVectorA = ReadVector<double>(kInputSize);
        const std::vector<double> kHostVectorB = ReadVector<double>(kInputSize);

        CudaVectorMin<double> cuda_vector_min(kInputSize);
        const std::vector<double> kResult =
            cuda_vector_min.Compute(kHostVectorA, kHostVectorB);

        PrintVector(kResult);
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        return 1;
    }

    return 0;
}
