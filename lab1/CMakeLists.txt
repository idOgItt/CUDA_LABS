cmake_minimum_required(VERSION 3.16)
project(SYCL_Test LANGUAGES CXX)

set(CMAKE_CXX_COMPILER "/opt/hipSYCL/ROCm/bin/acpp")
set(CMAKE_CXX_FLAGS "-std=c++17")

add_executable(sycl_test main.cpp)
