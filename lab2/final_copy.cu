#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

struct IOPaths {
    std::string inputFile;
    std::string outputFile;
};

namespace cuda_utils {
    inline void CheckCudaError(cudaError_t err, const char *msg) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
        }
    }

    class CudaArray {
    public:
        CudaArray(std::size_t width, std::size_t height, const cudaChannelFormatDesc &desc) {
            CheckCudaError(cudaMallocArray(&array_, &desc, width, height), "Failed to allocate cudaArray");
        }

        ~CudaArray() noexcept {
            if (array_) cudaFreeArray(array_);
        }

        cudaArray_t get() const noexcept { return array_; }

        void CopyFromHost(const void *hostData, std::size_t width, const std::size_t height,
                          const std::size_t pitch) const {
            CheckCudaError(cudaMemcpy2DToArray(array_, 0, 0, hostData, pitch, pitch, height, cudaMemcpyHostToDevice),
                           "Failed to copy image data to CUDA array");
        }

    private:
        cudaArray_t array_{nullptr};
    };

    class CudaTexture {
    public:
        CudaTexture(const cudaResourceDesc &res, const cudaTextureDesc &tex) {
            CheckCudaError(cudaCreateTextureObject(&texObj_, &res, &tex, NULL), "Failed to create texture object");
        }

        ~CudaTexture() noexcept {
            if (texObj_) cudaDestroyTextureObject(texObj_);
        }

        cudaTextureObject_t get() const noexcept { return texObj_; }

    private:
        cudaTextureObject_t texObj_{0};
    };

    template<typename T>
    class CudaBuffer {
    public:
        explicit CudaBuffer(const std::size_t count) {
            T *temp = nullptr;
            if (cudaMalloc(&temp, count * sizeof(T)) != cudaSuccess) {
                throw std::runtime_error("Failed to allocate CUDA memory");
            }
            buffer_.reset(temp);
        }

        T *Get() const noexcept { return buffer_.get(); }

        void CopyFromHost(const std::vector<T> &hostVector) {
            CopyToDevice(hostVector.data(), hostVector.size());
        }

        void CopyToHost(std::vector<T> &hostVector) {
            CopyFromDevice(hostVector.data(), hostVector.size());
        }

    private:
        struct CudaDeleter {
            void operator()(T *ptr) const noexcept {
                if (ptr) cudaFree(ptr);
            }
        };

        std::unique_ptr<T, CudaDeleter> buffer_;

        void CopyToDevice(const T *hostPtr, std::size_t size) {
            CheckCudaError(cudaMemcpy(buffer_.get(), hostPtr, size * sizeof(T), cudaMemcpyHostToDevice),
                           "Failed to copy to device");
        }

        void CopyFromDevice(T *hostPtr, std::size_t size) {
            CheckCudaError(cudaMemcpy(hostPtr, buffer_.get(), size * sizeof(T), cudaMemcpyDeviceToHost),
                           "Failed to copy from device");
        }
    };

    class CudaGraph {
    public:
        CudaGraph() {
            CheckCudaError(cudaStreamCreate(&stream_), "Failed to create stream");
        }

        ~CudaGraph() noexcept {
            if (graphExec_) cudaGraphExecDestroy(graphExec_);
            if (graph_) cudaGraphDestroy(graph_);
            cudaStreamDestroy(stream_);
        }

        template<typename Func>
        void Capture(Func launcher) {
            CheckCudaError(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
                           "Failed to begin graph capture");
            launcher(stream_);
            CheckCudaError(cudaStreamEndCapture(stream_, &graph_), "Failed to end graph capture");
            CheckCudaError(cudaGraphInstantiate(&graphExec_, graph_, NULL, NULL, 0), "Failed to instantiate graph");
        }

        void Launch() {
            CheckCudaError(cudaGraphLaunch(graphExec_, stream_), "Failed to launch graph");
            CheckCudaError(cudaStreamSynchronize(stream_), "Failed to synchronize stream");
        }

        cudaStream_t GetStream() const noexcept { return stream_; }

    private:
        cudaStream_t stream_{nullptr};
        cudaGraph_t graph_{nullptr};
        cudaGraphExec_t graphExec_{nullptr};
    };

    class CudaEvent {
    public:
        CudaEvent() {
            CheckCudaError(cudaEventCreate(&event_), "Failed to create CUDA event");
        }

        ~CudaEvent() noexcept {
            if (event_) cudaEventDestroy(event_);
        }

        cudaEvent_t get() const noexcept { return event_; }

    private:
        cudaEvent_t event_{nullptr};
    };
} // namespace cuda_utils

__global__ void sobelFilterKernel(uchar4 *outputImg, const std::size_t imgW, const std::size_t imgH,
                                  cudaTextureObject_t texObj) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr float kernelX[3][3] = {{-1.f, 0.f, 1.f}, {-2.f, 0.f, 2.f}, {-1.f, 0.f, 1.f}};
    constexpr float kernelY[3][3] = {{1.f, 2.f, 1.f}, {0.f, 0.f, 0.f}, {-1.f, -2.f, -1.f}};

    const int stepX = blockDim.x * gridDim.x;
    const int stepY = blockDim.y * gridDim.y;

    for (; row < imgH; row += stepY) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        for (; col < imgW; col += stepX) {
            float gradX = 0.f, gradY = 0.f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uchar4 pixel = tex2D<uchar4>(texObj, col + dx, row + dy);
                    float intensity = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
                    gradX += intensity * kernelX[dy + 1][dx + 1];
                    gradY += intensity * kernelY[dy + 1][dx + 1];
                }
            }
            float edgeValue = sqrtf(gradX * gradX + gradY * gradY);
            edgeValue = fminf(fmaxf(edgeValue, 0.f), 255.0001f);
            uchar4 centerPixel = tex2D<uchar4>(texObj, col, row);
            outputImg[row * imgW + col] = {
                static_cast<unsigned char>(edgeValue),
                static_cast<unsigned char>(edgeValue),
                static_cast<unsigned char>(edgeValue),
                centerPixel.w
            };
        }
    }
}

class SobelExecutor {
public:
    SobelExecutor(const uint32_t width, const uint32_t height, const std::vector<uchar4> &input)
        : width_(width), height_(height),
          numPixels_(static_cast<std::size_t>(width) * height),
          deviceOutput_(numPixels_),
          textureArray_(width, height, cudaCreateChannelDesc<uchar4>()),
          textureObject_(CreateTextureObject(textureArray_.get())) {
        textureArray_.CopyFromHost(input.data(), width, height, width * sizeof(uchar4));
        InitGraph();
    }

    std::vector<uchar4> Run() {
#ifdef ENABLE_TIMING
        const cuda_utils::CudaEvent start;
        const cuda_utils::CudaEvent stop;
        cuda_utils::CheckCudaError(cudaEventRecord(start.get(), graph_.GetStream()), "Failed to record start event");
#endif

        graph_.Launch();

#ifdef ENABLE_TIMING
        cuda_utils::CheckCudaError(cudaEventRecord(stop.get(), graph_.GetStream()), "Failed to record stop event");
        cuda_utils::CheckCudaError(cudaEventSynchronize(stop.get()), "Failed to sync stop event");

        [[maybe_unused]] float elapsedMs = 0.f;
        cuda_utils::CheckCudaError(cudaEventElapsedTime(&elapsedMs, start.get(), stop.get()), "Failed to measure time");
        std::cerr << "Execution time (ms): " << elapsedMs << std::endl;
#endif

        std::vector<uchar4> result(numPixels_);
        deviceOutput_.CopyToHost(result);
        return result;
    }

private:
    void InitGraph() {
        constexpr dim3 kThreads{32, 32};
        constexpr uint32_t kMaxGrid = 32;

        dim3 blocks((width_ + kThreads.x - 1) / kThreads.x,
                    (height_ + kThreads.y - 1) / kThreads.y);

        blocks.x = std::min(blocks.x, kMaxGrid);
        blocks.y = std::min(blocks.y, kMaxGrid);

        graph_.Capture([&](cudaStream_t stream) {
            sobelFilterKernel<<<blocks, kThreads, 0, stream>>>(deviceOutput_.Get(), width_, height_,
                                                               textureObject_.get());
        });
    }

    static cuda_utils::CudaTexture CreateTextureObject(const cudaArray_t array) {
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = array;

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        return cuda_utils::CudaTexture(resDesc, texDesc);
    }

    uint32_t width_, height_;
    std::size_t numPixels_;
    cuda_utils::CudaBuffer<uchar4> deviceOutput_;
    cuda_utils::CudaArray textureArray_;
    cuda_utils::CudaTexture textureObject_;
    cuda_utils::CudaGraph graph_;
};

void ReadImage(const std::string &path, uint32_t &width, uint32_t &height, std::vector<uchar4> &data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open input file: " + path);
    file.read(reinterpret_cast<char *>(&width), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&height), sizeof(uint32_t));
    const std::size_t numPixels = static_cast<std::size_t>(width) * height;
    data.resize(numPixels);
    file.read(reinterpret_cast<char *>(data.data()), numPixels * sizeof(uchar4));
}

void WriteImage(const std::string &path, const uint32_t width, const uint32_t height, const std::vector<uchar4> &data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open output file: " + path);
    file.write(reinterpret_cast<const char *>(&width), sizeof(uint32_t));
    file.write(reinterpret_cast<const char *>(&height), sizeof(uint32_t));
    file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(uchar4));
}

IOPaths ReadInputOutputPaths() {
    IOPaths paths;
    std::cin >> paths.inputFile >> paths.outputFile;
    return paths;
}


int main() {
    try {
        IOPaths paths = ReadInputOutputPaths();
        const std::string &inputFile = paths.inputFile;
        const std::string &outputFile = paths.outputFile;


        uint32_t width = 0, height = 0;
        std::vector<uchar4> imageData;
        ReadImage(inputFile, width, height, imageData);

        std::cerr << "Image size: " << width << " x " << height << std::endl;

        SobelExecutor executor(width, height, imageData);
        std::vector<uchar4> result = executor.Run();

        WriteImage(outputFile, width, height, result);
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
