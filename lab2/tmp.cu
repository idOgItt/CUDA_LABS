#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define PROCESS(expr) \
    e = expr; \
    if (e != cudaSuccess) { \
    std::cerr << cudaGetErrorString(e) << " " << __LINE__ << std::endl;\
    return 1; }


using uchar = unsigned char;

__global__ void sobel_kernel(uchar4 *output, size_t width, size_t height, cudaTextureObject_t tex) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr float gx_kernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    constexpr float gy_kernel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    const int x_dim = blockDim.x * gridDim.x;
    const int y_dim = blockDim.y * gridDim.y;


    for (; y < height; y += y_dim) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        for (; x < width; x += x_dim) {
            float gx = 0.0f, gy = 0.0f;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    auto pixel = tex2D<uchar4>(tex, x + dx, y + dy);
                    float gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

                    gx += gray * gx_kernel[dy + 1][dx + 1];
                    gy += gray * gy_kernel[dy + 1][dx + 1];
                }
            }

            float magnitude = sqrtf(gx * gx + gy * gy);
            magnitude = fminf(fmaxf(magnitude, 0.0f), 255.0001f);

            auto center = tex2D<uchar4>(tex, x, y);
            uchar4 result = {
                (uchar) magnitude,
                (uchar) magnitude,
                (uchar) magnitude,
                center.w
            };

            output[y * width + x] = result;
        }
    }
}

int main() {
    std::string input_path, output_path;
    std::cin >> input_path >> output_path;

    std::ifstream in_file(input_path, std::ios::binary);

    uint32_t width, height;
    in_file.read(reinterpret_cast<char *>(&width), sizeof(uint32_t));
    in_file.read(reinterpret_cast<char *>(&height), sizeof(uint32_t));

    std::cerr << width << " " << height << std::endl;
    cudaError_t e;

    size_t num_pixels = width * height;
    std::vector<uchar4> data(num_pixels, {0, 0, 0, 0});
    in_file.read(reinterpret_cast<char *>(data.data()), num_pixels * sizeof(uchar4));
    in_file.close();

    uchar4 *dev_output;
    auto desc = cudaCreateChannelDesc<uchar4>();
    cudaArray *arr;
    PROCESS(cudaMallocArray(&arr, &desc, width, height))
    PROCESS(
        cudaMemcpy2DToArray(arr, 0, 0, data.data(), width * sizeof(uchar4), width * sizeof(uchar4), height,
            cudaMemcpyHostToDevice))
    PROCESS(cudaMalloc(&dev_output, num_pixels * sizeof(uchar4)))

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    PROCESS(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    grid.x = std::min(grid.x, 32u);
    grid.y = std::min(grid.y, 32u);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sobel_kernel<<<grid, block>>>(dev_output, width, height, tex);

    cudaEventRecord(stop);

    std::vector<uchar4> out(num_pixels, {0, 0, 0, 0});
    PROCESS(cudaMemcpy(out.data(), dev_output, num_pixels * sizeof(uchar4), cudaMemcpyDeviceToHost));

    PROCESS(cudaGetLastError())

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cerr << "Milliseconds: " << milliseconds << std::endl;

    std::ofstream out_file(output_path, std::ios::binary);
    out_file.write(reinterpret_cast<char *>(&width), sizeof(uint32_t));
    out_file.write(reinterpret_cast<char *>(&height), sizeof(uint32_t));
    out_file.write(reinterpret_cast<char *>(out.data()), num_pixels * sizeof(uchar4));

    cudaDestroyTextureObject(tex);
    cudaFreeArray(arr);
    cudaFree(dev_output);

    return 0;
}
