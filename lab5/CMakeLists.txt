cmake_minimum_required(VERSION 3.16)

project(lab3 LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_HIP_ARCHITECTURES "gfx90c")

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc" CACHE STRING "HIP C++ Compiler" FORCE)
set(CMAKE_HIP_COMPILER "/opt/rocm/bin/hipcc" CACHE STRING "HIP Compiler" FORCE)

add_definitions(-D__HIP_PLATFORM_AMD__)
add_compile_definitions(ENABLE_TIMING)

enable_language(HIP)

find_package(HIP REQUIRED)
find_package(hipBLAS REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
        ${HIP_INCLUDE_DIRS}
        /opt/rocm/include
        /opt/rocm/hip/include
        /opt/cuda/targets/x86_64-linux/include
        ${OpenCV_INCLUDE_DIRS}
)

find_library(HIPRTC_LIB NAMES hiprtc HINTS /opt/rocm/lib)
find_library(HIPBLAS_LIB NAMES hipblas HINTS /opt/rocm/lib)

if (NOT HIPRTC_LIB)
    message(FATAL_ERROR "hiprtc не найден! Проверь установку ROCm.")
endif()

if (NOT HIPBLAS_LIB)
    message(FATAL_ERROR "hipBLAS не найден! Проверь установку ROCm.")
endif()

set(SOURCE_FILE ${CMAKE_SOURCE_DIR}/lab3.cu)
if (NOT EXISTS "${SOURCE_FILE}")
    message(FATAL_ERROR "Файл lab3.cu не найден! Проверьте путь.")
endif()

set(HIPIFIED_SRC "${CMAKE_BINARY_DIR}/lab3.cpp")
add_custom_command(
        OUTPUT ${HIPIFIED_SRC}
        COMMAND hipify-clang --cuda-path=/opt/cuda -I${OpenCV_INCLUDE_DIRS} ${SOURCE_FILE} -o ${HIPIFIED_SRC}
        DEPENDS ${SOURCE_FILE}
)

add_executable(lab3 ${HIPIFIED_SRC})
set_target_properties(lab3 PROPERTIES LINKER_LANGUAGE HIP)

target_link_libraries(lab3 PRIVATE ${HIPRTC_LIB} ${HIPBLAS_LIB} hip::host hip::device ${OpenCV_LIBS})
