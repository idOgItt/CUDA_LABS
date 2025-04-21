#include <iostream>
#include <vector>
#include <chrono>
#include <format>
#include <cmath>
#include <sycl/sycl.hpp>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>

constexpr double EPSILON = 1e-7;
const sycl::range<2> GLOBAL_RANGE(256, 256);
const sycl::range<2> LOCAL_RANGE(16, 16);
constexpr size_t WORKGROUP_SIZE = 256;
constexpr size_t GLOBAL_WORK_ITEMS = 2048;

uint32_t select_pivot_row(sycl::queue &queue,
                          sycl::buffer<double, 2> &matrix_buf,
                          uint32_t size, uint32_t col) {
    queue.wait();
    auto host_matrix = matrix_buf.get_host_access<sycl::access::mode::read>();
    uint32_t length = size - col;
    std::vector<double> abs_column(length);

    for (uint32_t row = col; row < size; ++row) {
        abs_column[row - col] = std::fabs(host_matrix[sycl::id{row, col}]);
    }

    thrust::device_vector<double> device_column(abs_column.begin(), abs_column.end());
    auto max_it = thrust::max_element(device_column.begin(), device_column.end());
    uint32_t offset = static_cast<uint32_t>(max_it - device_column.begin());

    if (device_column[offset] < EPSILON) {
        throw std::runtime_error("Matrix is singular or nearly singular");
    }
    return offset + col;
}

void swap_matrix_rows(sycl::queue &queue,
                      sycl::buffer<double, 2> &matrix_buf,
                      uint32_t size, uint32_t row1, uint32_t row2) {
    queue.submit([&](sycl::handler &cgh) {
        auto acc_mat = matrix_buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(sycl::nd_range<1>(GLOBAL_WORK_ITEMS, WORKGROUP_SIZE),
                         [=](sycl::nd_item<1> item) {
            size_t col_idx = item.get_global_id(0);
            while (col_idx <= size) {
                std::swap(acc_mat[sycl::id{row1, (uint32_t)col_idx}],
                          acc_mat[sycl::id{row2, (uint32_t)col_idx}]);
                col_idx += item.get_global_range(0);
            }
        });
    });
}

void eliminate_below_pivot(sycl::queue &queue,
                            sycl::buffer<double, 2> &matrix_buf,
                            uint32_t size, uint32_t col) {
    queue.submit([&](sycl::handler &cgh) {
        auto acc_mat = matrix_buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(sycl::nd_range<2>(GLOBAL_RANGE, LOCAL_RANGE),
                         [=](sycl::nd_item<2> item) {
            uint32_t row = col + 1 + item.get_global_id(0);
            uint32_t c = col + item.get_global_id(1);
            double pivot_val = acc_mat[sycl::id{col, col}];
            while (row < size) {
                double factor = acc_mat[sycl::id{row, col}] / pivot_val;
                uint32_t inner_col = c;
                while (inner_col <= size) {
                    if (inner_col != col) {
                        acc_mat[sycl::id{row, inner_col}] -=
                            factor * acc_mat[sycl::id{col, inner_col}];
                    }
                    inner_col += item.get_global_range(1);
                }
                row += item.get_global_range(0);
            }
        });
    });

    queue.submit([&](sycl::handler &cgh) {
        auto acc_mat = matrix_buf.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::nd_range<1>(GLOBAL_WORK_ITEMS, WORKGROUP_SIZE),
                         [=](sycl::nd_item<1> item) {
            uint32_t row = col + 1 + item.get_global_id(0);
            while (row < size) {
                acc_mat[sycl::id{row, col}] = 0.0;
                row += item.get_global_range(0);
            }
        });
    });
}

void eliminate_above_pivot(sycl::queue &queue,
                            sycl::buffer<double, 2> &matrix_buf,
                            uint32_t size, uint32_t col) {
    queue.submit([&](sycl::handler &cgh) {
        auto acc_mat = matrix_buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(sycl::nd_range<2>(GLOBAL_RANGE, LOCAL_RANGE),
                         [=](sycl::nd_item<2> item) {
            uint32_t row = item.get_global_id(0);
            uint32_t c = col + item.get_global_id(1);
            double pivot_val = acc_mat[sycl::id{col, col}];
            while (row < col) {
                double factor = acc_mat[sycl::id{row, col}] / pivot_val;
                uint32_t inner_col = c;
                while (inner_col <= size) {
                    if (inner_col != col) {
                        acc_mat[sycl::id{row, inner_col}] -=
                            factor * acc_mat[sycl::id{col, inner_col}];
                    }
                    inner_col += item.get_global_range(1);
                }
                row += item.get_global_range(0);
            }
        });
    });

    queue.submit([&](sycl::handler &cgh) {
        auto acc_mat = matrix_buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(sycl::nd_range<1>(GLOBAL_WORK_ITEMS, WORKGROUP_SIZE),
                         [=](sycl::nd_item<1> item) {
            uint32_t row = item.get_global_id(0);
            while (row < col) {
                acc_mat[sycl::id{row, col}] = 0.0;
                row += item.get_global_range(0);
            }
        });
    });
}

void normalize_rows(sycl::queue &queue,
                    sycl::buffer<double, 2> &matrix_buf,
                    uint32_t size) {
    queue.submit([&](sycl::handler &cgh) {
        auto acc_mat = matrix_buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(sycl::nd_range<2>(GLOBAL_RANGE, LOCAL_RANGE),
                         [=](sycl::nd_item<2> item) {
            uint32_t row = item.get_global_id(0);
            uint32_t col_idx = item.get_global_id(1);
            while (row < size) {
                double diag = acc_mat[sycl::id{row, row}];
                uint32_t inner_col = col_idx;
                while (inner_col <= size) {
                    acc_mat[sycl::id{row, inner_col}] /= diag;
                    inner_col += item.get_global_range(1);
                }
                row += item.get_global_range(0);
            }
        });
    });
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    uint32_t size;
    std::cin >> size;

    std::vector<double> data(size * (size + 1));
    for (uint32_t row = 0; row < size; ++row) {
        for (uint32_t col_idx = 0; col_idx < size; ++col_idx) {
            std::cin >> data[row * (size + 1) + col_idx];
        }
        std::cin >> data[row * (size + 1) + size];
    }

    sycl::queue queue(sycl::default_selector_v,
                      sycl::property::queue::enable_profiling{});
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        sycl::buffer<double, 2> matrix_buf(data.data(), sycl::range<2>(size, size + 1));

        for (uint32_t col = 0; col < size; ++col) {
            uint32_t pivot_row = select_pivot_row(queue, matrix_buf, size, col);
            if (pivot_row != col)
                swap_matrix_rows(queue, matrix_buf, size, pivot_row, col);
            eliminate_below_pivot(queue, matrix_buf, size, col);
        }
        for (int32_t col = size - 1; col >= 0; --col) {
            eliminate_above_pivot(queue, matrix_buf, size, col);
        }
        normalize_rows(queue, matrix_buf, size);

    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    queue.wait();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cerr << "Elapsed (ms): " << elapsed_ms << "\n";

    for (uint32_t row = 0; row < size; ++row) {
        std::cout << std::format("{:.10e}", data[row * (size + 1) + size]);
        if (row + 1 < size) std::cout << " ";
    }
    std::cout << "\n";

    return 0;
}
