cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cuda_utils)

find_package(Torch REQUIRED)

add_executable(cuda_utils example.cpp)

target_link_libraries(cuda_utils "${TORCH_LIBRARIES}")
set_property(TARGET cuda_utils PROPERTY CXX_STANDARD 14)
