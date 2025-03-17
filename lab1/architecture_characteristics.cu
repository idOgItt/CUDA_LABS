#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <thread>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <sys/statvfs.h>

void PrintGPUProperties() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    std::cout << "GPU Name: " << deviceProp.name << std::endl;
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared Memory Per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Constant Memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
    std::cout << "Registers Per Block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
}

void PrintCPUProperties() {
    std::cout << "CPU Cores: " << std::thread::hardware_concurrency() << std::endl;
    long pageSize = sysconf(_SC_PAGESIZE);
    long totalRAM = sysconf(_SC_PHYS_PAGES) * pageSize;
    std::cout << "Total RAM: " << totalRAM / (1024 * 1024) << " MB" << std::endl;
}

void PrintDiskInfo() {
    struct statvfs stat;
    if (statvfs("/", &stat) != 0) {
        std::cerr << "Error retrieving disk information" << std::endl;
        return;
    }
    unsigned long long totalDisk = stat.f_blocks * stat.f_frsize;
    std::cout << "Total Disk Space: " << totalDisk / (1024 * 1024 * 1024) << " GB" << std::endl;
}

void PrintSystemInfo() {
#if defined(_WIN32) || defined(_WIN64)
    std::cout << "Operating System: Windows" << std::endl;
#elif defined(__linux__)
    std::cout << "Operating System: Linux" << std::endl;
#elif defined(__APPLE__)
    std::cout << "Operating System: macOS" << std::endl;
#endif

#ifdef __GNUC__
    std::cout << "Compiler: GCC " << __GNUC__ << "." << __GNUC_MINOR__ << std::endl;
#elif defined(__clang__)
    std::cout << "Compiler: Clang " << __clang_major__ << "." << __clang_minor__ << std::endl;
#endif
}


int main() {
    PrintGPUProperties();
    PrintCPUProperties();
    PrintDiskInfo();
    PrintSystemInfo();
    return 0;
}
