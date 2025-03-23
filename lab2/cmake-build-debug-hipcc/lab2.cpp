#include <hip/hip_runtime.h>


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
    inline void CheckCudaError(hipError_t err, const char *msg) {
        if (err != hipSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
        }
    }

    class CudaArray {
    public:
        CudaArray(std::size_t width, std::size_t height, const hipChannelFormatDesc &desc) {
            CheckCudaError(hipMallocArray(&array_, &desc, width, height), "Failed to allocate hipArray");
        }

        ~CudaArray() noexcept {
            if (array_) hipFreeArray(array_);
        }

        hipArray_t get() const noexcept { return array_; }

        void CopyFromHost(const void *hostData, std::size_t width, const std::size_t height,
                          const std::size_t pitch) const {
            CheckCudaError(hipMemcpy2DToArray(array_, 0, 0, hostData, pitch, pitch, height, hipMemcpyHostToDevice),
                           "Failed to copy image data to CUDA array");
        }

    private:
        hipArray_t array_{nullptr};
    };

    class CudaTexture {
    public:
        CudaTexture(const hipResourceDesc &res, const hipTextureDesc &tex) {
            CheckCudaError(hipCreateTextureObject(&texObj_, &res, &tex, NULL), "Failed to create texture object");
        }

        ~CudaTexture() noexcept {
            if (texObj_) hipDestroyTextureObject(texObj_);
        }

        hipTextureObject_t get() const noexcept { return texObj_; }

    private:
        hipTextureObject_t texObj_{0};
    };

    template<typename T>
    class CudaBuffer {
    public:
        explicit CudaBuffer(const std::size_t count) {
            T *temp = nullptr;
            if (hipMalloc(&temp, count * sizeof(T)) != hipSuccess) {
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
                if (ptr) hipFree(ptr);
            }
        };

        std::unique_ptr<T, CudaDeleter> buffer_;

        void CopyToDevice(const T *hostPtr, std::size_t size) {
            CheckCudaError(hipMemcpy(buffer_.get(), hostPtr, size * sizeof(T), hipMemcpyHostToDevice),
                           "Failed to copy to device");
        }

        void CopyFromDevice(T *hostPtr, std::size_t size) {
            CheckCudaError(hipMemcpy(hostPtr, buffer_.get(), size * sizeof(T), hipMemcpyDeviceToHost),
                           "Failed to copy from device");
        }
    };

    class CudaGraph {
    public:
        CudaGraph() {
            CheckCudaError(hipStreamCreate(&stream_), "Failed to create stream");
        }

        ~CudaGraph() noexcept {
            if (graphExec_) hipGraphExecDestroy(graphExec_);
            if (graph_) hipGraphDestroy(graph_);
            hipStreamDestroy(stream_);
        }

        template<typename Func>
        void Capture(Func launcher) {
            CheckCudaError(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal),
                           "Failed to begin graph capture");
            launcher(stream_);
            CheckCudaError(hipStreamEndCapture(stream_, &graph_), "Failed to end graph capture");
            CheckCudaError(hipGraphInstantiate(&graphExec_, graph_, NULL, NULL, 0), "Failed to instantiate graph");
        }

        void Launch() {
            CheckCudaError(hipGraphLaunch(graphExec_, stream_), "Failed to launch graph");
            CheckCudaError(hipStreamSynchronize(stream_), "Failed to synchronize stream");
        }

        hipStream_t GetStream() const noexcept { return stream_; }

    private:
        hipStream_t stream_{nullptr};
        hipGraph_t graph_{nullptr};
        hipGraphExec_t graphExec_{nullptr};
    };

    class CudaEvent {
    public:
        CudaEvent() {
            CheckCudaError(hipEventCreate(&event_), "Failed to create CUDA event");
        }

        ~CudaEvent() noexcept {
            if (event_) hipEventDestroy(event_);
        }

        hipEvent_t get() const noexcept { return event_; }

    private:
        hipEvent_t event_{nullptr};
    };
} // namespace cuda_utils

__global__ void sobelFilterKernel(uchar4 *outputImg, const std::size_t imgW, const std::size_t imgH,
                                  hipTextureObject_t texObj) {
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
          textureArray_(width, height, hipCreateChannelDesc<uchar4>()),
          textureObject_(CreateTextureObject(textureArray_.get())) {
        textureArray_.CopyFromHost(input.data(), width, height, width * sizeof(uchar4));
        InitGraph();
    }

    std::vector<uchar4> Run() {
#ifdef ENABLE_TIMING
        const cuda_utils::CudaEvent start;
        const cuda_utils::CudaEvent stop;
        cuda_utils::CheckCudaError(hipEventRecord(start.get(), graph_.GetStream()), "Failed to record start event");
#endif

        graph_.Launch();

#ifdef ENABLE_TIMING
        cuda_utils::CheckCudaError(hipEventRecord(stop.get(), graph_.GetStream()), "Failed to record stop event");
        cuda_utils::CheckCudaError(hipEventSynchronize(stop.get()), "Failed to sync stop event");

        [[maybe_unused]] float elapsedMs = 0.f;
        cuda_utils::CheckCudaError(hipEventElapsedTime(&elapsedMs, start.get(), stop.get()), "Failed to measure time");
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

        graph_.Capture([&](hipStream_t stream) {
            sobelFilterKernel<<<blocks, kThreads, 0, stream>>>(deviceOutput_.Get(), width_, height_,
                                                               textureObject_.get());
        });
    }

    static cuda_utils::CudaTexture CreateTextureObject(const hipArray_t array) {
        hipResourceDesc resDesc{};
        resDesc.resType = hipResourceTypeArray;
        resDesc.res.array.array = array;

        hipTextureDesc texDesc{};
        texDesc.addressMode[0] = hipAddressModeClamp;
        texDesc.addressMode[1] = hipAddressModeClamp;
        texDesc.filterMode = hipFilterModePoint;
        texDesc.readMode = hipReadModeElementType;
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
