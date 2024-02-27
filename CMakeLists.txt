cmake_minimum_required(VERSION 3.10)
project(EasyNN)


#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)  # 将 C++ 标准设置为 C++ 11
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # C++ 11 是强制要求，不会衰退至低版本
set(CMAKE_CXX_EXTENSIONS OFF)  # 禁止使用编译器特有扩展



option(EASTNN_USE_CUDA "enable cuda " ON)

if(EASTNN_USE_CUDA)
    add_definitions(-DEASTNN_USE_CUDA=1)  
    enable_language(CUDA)
endif()

add_executable(01_gemm_naive src/gemm/01_gemm_naive.cu)
add_executable(02_gemm_sharemem src/gemm/02_gemm_sharemem.cu)
add_executable(03_gemm src/gemm/03_gemm.cu)
#add_subdirectory(example)