#include <hip/hip_runtime.h>


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

__global__ void VectorPerElemMinDouble(const double* first_vector,
                                       const double* second_vector,
                                       double* result,
                                       const int32_t* data_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t size = *data_size; idx < size; idx += stride) {
        result[idx] = fmin(first_vector[idx], second_vector[idx]);
    }
}

class CudaGraph {
public:
    CudaGraph() { hipStreamCreate(&stream_); }

    ~CudaGraph() {
        if (graphExec_) hipGraphExecDestroy(graphExec_);
        if (graph_) hipGraphDestroy(graph_);
        hipStreamDestroy(stream_);
    }

    void Capture(std::function<void(hipStream_t)> kernel_launcher) {
        hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal);
        kernel_launcher(stream_);
        hipStreamEndCapture(stream_, &graph_);
        hipGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0);
    }

    void Launch() {
        hipGraphLaunch(graphExec_, stream_);
        hipDeviceSynchronize();
    }

private:
    hipStream_t stream_;
    hipGraph_t graph_{nullptr};
    hipGraphExec_t graphExec_{nullptr};
};

template <typename T,
          typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
class CudaBuffer {
public:
    explicit CudaBuffer(const std::size_t kSize) { Allocate(kSize); }

    T* Get() const noexcept { return buffer_.get(); }

    void CopyFromHost(const std::vector<T>& host_vector) {
        CopyToDevice(host_vector.data(), host_vector.size());
    }

    void CopyToHost(std::vector<T>& host_vector) {
        CopyToHost(host_vector.data(), host_vector.size());
    }

private:
    struct CudaDeleter {
        void operator()(T* ptr) const noexcept {
            if (ptr) {
                hipFree(ptr);
            }
        }
    };

    std::unique_ptr<T, CudaDeleter> buffer_;

    void Allocate(const std::size_t kSize) {
        T* temp_ptr = nullptr;
        if (hipMalloc(&temp_ptr, kSize * sizeof(T)) != hipSuccess) {
            throw std::runtime_error("Failed to allocate CUDA memory");
        }
        buffer_.reset(temp_ptr);
    }

    void CopyToDevice(const T* host_ptr, std::size_t size) {
        hipMemcpy(buffer_.get(), host_ptr, size * sizeof(T),
                   hipMemcpyHostToDevice);
    }

    void CopyToHost(T* host_ptr, std::size_t size) {
        hipMemcpy(host_ptr, buffer_.get(), size * sizeof(T),
                   hipMemcpyDeviceToHost);
    }
};

template <typename T,
          typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
class CudaVectorMin {
public:
    explicit CudaVectorMin(int32_t size)
        : vector_size_(size),
          device_vector_a_(size),
          device_vector_b_(size),
          device_result_(size),
          device_data_size_(1) {
        Init();
    }

    std::vector<T> Compute(const std::vector<T>& host_vector_a,
                           const std::vector<T>& host_vector_b) {
        device_vector_a_.CopyFromHost(host_vector_a);
        device_vector_b_.CopyFromHost(host_vector_b);
        graph_.Launch();
        std::vector<T> host_result(vector_size_);
        device_result_.CopyToHost(host_result);
        return host_result;
    }

private:
    int32_t vector_size_;
    int32_t numSM_{};
    CudaBuffer<T> device_vector_a_;
    CudaBuffer<T> device_vector_b_;
    CudaBuffer<T> device_result_;
    CudaBuffer<int32_t> device_data_size_;
    CudaGraph graph_;

    void Init() {
        GetDeviceProperties();
        device_data_size_.CopyFromHost(std::vector<int32_t>{vector_size_});
        CaptureCudaGraph();
    }

    void GetDeviceProperties() {
        hipDeviceProp_t device_prop;
        hipGetDeviceProperties(&device_prop, 0);
        numSM_ = device_prop.multiProcessorCount;
    }

    void CaptureCudaGraph() {
        int32_t grid_size = numSM_ * 4;
        graph_.Capture([&](hipStream_t stream) {
            VectorPerElemMinDouble<<<grid_size, kThreadsPerBlock, 0, stream>>>(
                device_vector_a_.Get(), device_vector_b_.Get(),
                device_result_.Get(), device_data_size_.Get());
        });
    }
};

template <typename T,
          typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
std::vector<T> ReadVector(const int32_t kSize) {
    std::vector<T> vector(kSize);
    for (T& value : vector) {
        std::cin >> value;
    }
    return vector;
}

template <typename T,
          typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
void PrintVector(const std::vector<T>& vec) {
    constexpr int32_t kPrecision = 10;
    std::cout.precision(kPrecision);
    std::cout << std::scientific;

    std::copy(vec.begin(), vec.end() - 1,
              std::ostream_iterator<T>(std::cout, " "));
    std::cout << vec.back() << std::endl;
}

int32_t ReadInputSize() {
    int32_t input_size;
    std::cin >> input_size;

    constexpr int32_t kMaxSize = (1 << 25);
    if (input_size <= 0 || input_size >= kMaxSize) {
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
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
