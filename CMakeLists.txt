cmake_minimum_required(VERSION 3.10)
project(EasyNN)


set(CMAKE_CXX_STANDARD 11)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)  
set(CMAKE_CXX_EXTENSIONS OFF)  



option(EASTNN_USE_CUDA "enable cuda " ON)

if(EASTNN_USE_CUDA)
    add_definitions(-DEASTNN_USE_CUDA=1)  
    enable_language(CUDA)
endif()

add_executable(01_add src/add/01_add.cu)
add_executable(02_add src/add/02_add.cu)

add_executable(01_reduce src/reduce/01_reduce.cu)
add_executable(03_reduce src/reduce/03_reduce.cu)


add_executable(01_gemm_naive src/gemm/01_gemm_naive.cu)
add_executable(02_gemm_sharemem src/gemm/02_gemm_sharemem.cu)
add_executable(03_gemm src/gemm/03_gemm.cu)
add_executable(02_my_gemm_share src/gemm/02_my_gemm_share.cu)
add_executable(gemm_test src/gemm/gemm_test.cu)
add_executable(cuda_thread src/cuda_thread.cu)
#add_subdirectory(example)