cmake_minimum_required(VERSION 3.16)

project(lab1 LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_HIP_ARCHITECTURES "gfx90c")

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc" CACHE STRING "HIP C++ Compiler" FORCE)
set(CMAKE_HIP_COMPILER "/opt/rocm/bin/hipcc" CACHE STRING "HIP Compiler" FORCE)

add_definitions(-D__HIP_PLATFORM_AMD__)

enable_language(HIP)

find_package(HIP REQUIRED)
find_package(hipBLAS REQUIRED)

# Добавляем пути к заголовкам
include_directories(
        ${HIP_INCLUDE_DIRS}
        /opt/rocm/include
        /opt/rocm/hip/include
        /opt/cuda/targets/x86_64-linux/include
)

# Поиск библиотек HIP
find_library(HIPRTC_LIB NAMES hiprtc HINTS /opt/rocm/lib)
find_library(HIPBLAS_LIB NAMES hipblas HINTS /opt/rocm/lib)

if (NOT HIPRTC_LIB)
    message(FATAL_ERROR "hiprtc не найден! Проверь установку ROCm.")
endif()

if (NOT HIPBLAS_LIB)
    message(FATAL_ERROR "hipBLAS не найден! Проверь установку ROCm.")
endif()

# Проверяем наличие исходного файла
set(SOURCE_FILE ${CMAKE_SOURCE_DIR}/lab1_final.cu)
if (NOT EXISTS "${SOURCE_FILE}")
    message(FATAL_ERROR "Файл lab1_final.cu не найден! Проверьте путь.")
endif()

# Автоматическая конвертация CUDA -> HIP
set(HIPIFIED_SRC "${CMAKE_BINARY_DIR}/hip_code.cpp")
add_custom_command(
        OUTPUT ${HIPIFIED_SRC}
        COMMAND hipify-clang --cuda-path=/opt/cuda ${SOURCE_FILE} -o ${HIPIFIED_SRC}
        DEPENDS ${SOURCE_FILE}
)

# Создаём исполняемый файл и указываем hipcc как компилятор
add_executable(lab1 ${HIPIFIED_SRC})
set_target_properties(lab1 PROPERTIES LINKER_LANGUAGE HIP)

# Линкуем HIP и hipBLAS
target_link_libraries(lab1 PRIVATE ${HIPRTC_LIB} ${HIPBLAS_LIB} hip::host hip::device)
