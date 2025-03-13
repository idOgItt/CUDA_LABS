#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>
#include <iostream>

int main() {
    try {
        sycl::queue q;
        std::cout << "Running on: "
                  << q.get_device().template get_info<sycl::info::device::name>()
                  << std::endl;
    } catch (sycl::exception const &e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
