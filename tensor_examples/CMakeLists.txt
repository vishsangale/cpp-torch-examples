cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(tensor_examples)

find_package(Torch REQUIRED)

add_executable(tensor_examples example.cpp)

target_link_libraries(tensor_examples "${TORCH_LIBRARIES}")
set_property(TARGET tensor_examples PROPERTY CXX_STANDARD 14)
