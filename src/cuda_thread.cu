#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>


__global__ void cuda_thread_idx()
{
    if(blockIdx.x ==0 && blockIdx.y ==0 && threadIdx.x ==0 && threadIdx.y == 0)
    {    
        printf("blockIdx.z = %d  ,threadIdx.z =%d \n",blockIdx.z,threadIdx.z);
        printf("blockDim.x = %d,blockDim.y = %d,blockDim.z = %d \n",blockDim.x,blockDim.y,blockDim.z);
        printf("gridDim.x = %d,gridDim.y = %d,gridDim.z = %d \n",gridDim.x,gridDim.y,gridDim.z);
    }

    //计算线程的全局索引
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    size_t global_idx = by*(gridDim.x*blockDim.x*blockDim.y)+bx*(blockDim.x*blockDim.y)+ty*blockDim.x+tx;
    printf("global_idx = %ld  \n",global_idx);

}



int main()
{

    dim3 block(3,2,1);
    dim3 grid(5,6,1);
    cuda_thread_idx<<<grid,block>>>();
    cudaDeviceSynchronize();
    printf("cuda_thread run done!\n");

    return 0;
}