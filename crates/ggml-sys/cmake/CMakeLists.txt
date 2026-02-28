cmake_minimum_required(VERSION 3.14)
project("ggml-wrapper" C CXX)

set(GGML_SOURCE_DIR "" CACHE PATH "Path to ggml source")
add_subdirectory(${GGML_SOURCE_DIR} ggml)
