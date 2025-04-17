#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/iterator/transform_iterator.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

constexpr int    BLOCK_SIZE = 256;

struct AbsOp {
    __host__ __device__
    double operator()(double x) const {
        return x < 0 ? -x : x;
    }
};

__global__ void swap_rows(double* A, double* b, int n, int r1, int r2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double tmp = A[r1*n + j];
        A[r1*n + j] = A[r2*n + j];
        A[r2*n + j] = tmp;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double tb = b[r1];
        b[r1] = b[r2];
        b[r2] = tb;
    }
}

__global__ void eliminate_kernel(double* A, double* b, int n, int k, double pivotVal) {
    int row = blockIdx.y + k + 1;
    if (row >= n) return;
    int col = blockIdx.x * blockDim.x + threadIdx.x + k;
    double factor = A[row*n + k] / pivotVal;
    if (col < n) {
        A[row*n + col] -= factor * A[k*n + col];
    }
    if (col == k) {
        b[row] -= factor * b[k];
    }
}

void set_binary_mode() {
#ifdef _WIN32
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
}

int main() {
    set_binary_mode();

    int n;
    std::cin.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<double> h_A(n * n), h_b(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            std::cin.read(reinterpret_cast<char*>(&h_A[i*n + j]), sizeof(double));
    for (int i = 0; i < n; ++i)
        std::cin.read(reinterpret_cast<char*>(&h_b[i]), sizeof(double));

    double *d_A, *d_b;
    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMalloc(&d_b, n       * sizeof(double));
    cudaMemcpy(d_A, h_A.data(), n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), n  *sizeof(double), cudaMemcpyHostToDevice);

    dim3 block1(BLOCK_SIZE);
    for (int k = 0; k < n; ++k) {
        thrust::device_ptr<double> col_ptr(d_A + k*n + k);
        auto abs_begin = thrust::make_transform_iterator(col_ptr, AbsOp());
        auto abs_end   = abs_begin + (n - k);
        auto max_it    = thrust::max_element(abs_begin, abs_end);
        int offset     = max_it - abs_begin;
        int p          = k + offset;

        if (p != k) {
            int swap_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            swap_rows<<<swap_blocks, block1>>>(d_A, d_b, n, k, p);
            cudaDeviceSynchronize();
        }

        double pivotVal;
        cudaMemcpy(&pivotVal, d_A + k*n + k, sizeof(double), cudaMemcpyDeviceToHost);

        int rows = n - k - 1;
        if (rows > 0) {
            dim3 grid((n - k + BLOCK_SIZE - 1) / BLOCK_SIZE, rows);
            eliminate_kernel<<<grid, block1>>>(d_A, d_b, n, k, pivotVal);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(h_A.data(), d_A, n*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b.data(), d_b, n  *sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);

    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < n; ++j) {
            sum += h_A[i*n + j] * x[j];
        }
        x[i] = (h_b[i] - sum) / h_A[i*n + i];
    }

    std::cout << std::scientific << std::setprecision(10);
    for (int i = 0; i < n; ++i) {
        std::cout << x[i] << (i + 1 < n ? ' ' : '\n');
    }
    return 0;
}
