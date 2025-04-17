#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct IOPaths {
    std::string inputFile;
    std::string outputFile;
};

namespace cuda_utils {
    inline void Check(cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
        }
    }

    template <typename T>
    class CudaBuffer {
    public:
        explicit CudaBuffer(size_t count) {
            T* ptr;
            cuda_utils::Check(cudaMalloc(&ptr, count * sizeof(T)), "Failed to allocate device buffer");
            ptr_.reset(ptr);
        }

        T* Get() const noexcept { return ptr_.get(); }

        void CopyFromHost(const std::vector<T>& vec) {
            cuda_utils::Check(cudaMemcpy(ptr_.get(), vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice),
                              "Failed to copy to device");
        }

        void CopyToHost(std::vector<T>& vec) const {
            cuda_utils::Check(cudaMemcpy(vec.data(), ptr_.get(), vec.size() * sizeof(T), cudaMemcpyDeviceToHost),
                              "Failed to copy from device");
        }

    private:
        struct Deleter {
            void operator()(T* p) const noexcept {
                if (p) cudaFree(p);
            }
        };
        std::unique_ptr<T, Deleter> ptr_;
    };

    class CudaGraph {
    public:
        CudaGraph() { Check(cudaStreamCreate(&stream_), "Stream creation failed"); }

        ~CudaGraph() {
            if (exec_) cudaGraphExecDestroy(exec_);
            if (graph_) cudaGraphDestroy(graph_);
            cudaStreamDestroy(stream_);
        }

        template <typename Func>
        void Capture(Func&& launcher) {
            Check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), "Begin capture failed");
            launcher(stream_);
            Check(cudaStreamEndCapture(stream_, &graph_), "End capture failed");
            Check(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0), "Graph instantiation failed");
        }

        void Launch() {
            Check(cudaGraphLaunch(exec_, stream_), "Graph launch failed");
            Check(cudaStreamSynchronize(stream_), "Stream sync failed");
        }

        cudaStream_t GetStream() const noexcept { return stream_; }

    private:
        cudaStream_t stream_{};
        cudaGraph_t graph_{};
        cudaGraphExec_t exec_{};
    };

    class CudaEvent {
    public:
        CudaEvent() { Check(cudaEventCreate(&event_), "Event creation failed"); }
        ~CudaEvent() { cudaEventDestroy(event_); }

        cudaEvent_t Get() const noexcept { return event_; }

    private:
        cudaEvent_t event_{};
    };
} // namespace cuda_utils

__global__ void classifyKernel(uchar4* image, int width, int height, const float3* classes, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    for (; idx < total; idx += blockDim.x * gridDim.x) {
        uchar4 pix = image[idx];
        float3 vec = make_float3(pix.x, pix.y, pix.z);
        float len = sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        if (len > 1e-5f) {
            vec.x /= len;
            vec.y /= len;
            vec.z /= len;
        }


        float maxDot = -1.0f;
        int bestClass = 0;
        for (int i = 0; i < numClasses; ++i) {
            float3 cl = classes[i];
            float dot = vec.x * cl.x + vec.y * cl.y + vec.z * cl.z;
            if (dot > maxDot) {
                maxDot = dot;
                bestClass = i;
            }
        }
        image[idx].w = bestClass;
    }
}

class Classifier {
public:
    Classifier(uint32_t w, uint32_t h, const std::vector<uchar4>& input, const std::vector<float3>& classes)
        : width_(w), height_(h), numPixels_(w * h),
          imageDevice_(numPixels_), classDevice_(classes.size()), classes_(classes) {
        imageDevice_.CopyFromHost(input);
        classDevice_.CopyFromHost(classes);
        InitGraph();
    }

    std::vector<uchar4> Run() {
        cuda_utils::CudaEvent start, stop;
        cudaEventRecord(start.Get(), graph_.GetStream());

        graph_.Launch();

        cudaEventRecord(stop.Get(), graph_.GetStream());
        cudaEventSynchronize(stop.Get());
        float ms = 0;
        cudaEventElapsedTime(&ms, start.Get(), stop.Get());
        std::cerr << "Execution time (ms): " << ms << '\n';

        std::vector<uchar4> output(numPixels_);
        imageDevice_.CopyToHost(output);
        return output;
    }

private:
    void InitGraph() {
        constexpr int blockSize = 256;
        constexpr int gridSize = 256;

        graph_.Capture([&](cudaStream_t stream) {
            classifyKernel<<<gridSize, blockSize, 0, stream>>>(
                imageDevice_.Get(), width_, height_, classDevice_.Get(), static_cast<int>(classes_.size()));
        });
    }

    uint32_t width_, height_;
    size_t numPixels_;
    std::vector<float3> classes_;
    cuda_utils::CudaBuffer<uchar4> imageDevice_;
    cuda_utils::CudaBuffer<float3> classDevice_;
    cuda_utils::CudaGraph graph_;
};

void ReadImage(const std::string& path, uint32_t& width, uint32_t& height, std::vector<uchar4>& data) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open input file");
    in.read(reinterpret_cast<char*>(&width), sizeof(width));
    in.read(reinterpret_cast<char*>(&height), sizeof(height));
    data.resize(static_cast<size_t>(width) * height);
    in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(uchar4));
}

void WriteImage(const std::string& path, uint32_t width, uint32_t height, const std::vector<uchar4>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open output file");
    out.write(reinterpret_cast<const char*>(&width), sizeof(width));
    out.write(reinterpret_cast<const char*>(&height), sizeof(height));
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uchar4));
}

IOPaths ReadPaths() {
    IOPaths paths;
    std::cin >> paths.inputFile >> paths.outputFile;
    return paths;
}

std::vector<float3> ReadClasses(const std::vector<uchar4>& data, uint32_t width, uint32_t height) {
    size_t NC;
    std::cin >> NC;
    std::vector<float3> classes(32);

    for (size_t i = 0; i < NC; ++i) {
        size_t NP;
        std::cin >> NP;
        float3 sum{0.f, 0.f, 0.f};

        for (size_t j = 0; j < NP; ++j) {
            size_t x, y;
            std::cin >> x >> y;

            if (x >= width || y >= height) {
                throw std::runtime_error("Class input contains out-of-bounds coordinate");
            }

            uchar4 pix = data[y * width + x];
            sum.x += pix.x;
            sum.y += pix.y;
            sum.z += pix.z;
        }

        sum.x /= NP; sum.y /= NP; sum.z /= NP;
        float len = std::sqrt(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
        if (len > 1e-5f) {
            sum.x /= len;
            sum.y /= len;
            sum.z /= len;
        }

        classes[i] = sum;
    }

    return classes;
}


int main() {
    try {
        IOPaths paths = ReadPaths();

        uint32_t width = 0, height = 0;
        std::vector<uchar4> image;
        ReadImage(paths.inputFile, width, height, image);
        std::cerr << "Loaded image: " << width << "x" << height << '\n';

        std::vector<float3> classes = ReadClasses(image, width, height);

        Classifier classifier(width, height, image, classes);
        std::vector<uchar4> result = classifier.Run();

        WriteImage(paths.outputFile, width, height, result);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
