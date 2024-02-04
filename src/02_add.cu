#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include"ultic.h"
#include"benchmark.h"

#define N 100000000

void cpu_add(const float * x, const float * y, float *z,size_t n)
{
    for(int i=0;i<n;i++)
    {
        z[i] = x[i] + y[i] ;
    }
}

void init_array(float * array,float x,int n)
{
    for(int i=0;i<n;i++)
    {
        array[i] = x;
    }
}

void compare_array(const float *x,const float * y, int n)
{

    for(int i=0;i<N;i++)
    {
        if(x[i]-y[i]>0.01)
        {
            printf("faiulre \n");
            break;
        }
    }
}

__global__ void cuda_add(const float * x, const float * y, float *z,size_t n)
{
    // int block_idx = blockIdx.x;
    // int thread_idx = threadIdx.x;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(idx<N)
    {
        z[idx] = x[idx]+y[idx];
    }

}



int main()
{
    float *hx,*hy,*hz,*dx,*dy,*dz;
    float *res;
    size_t nbytes = N*sizeof(float);

    hx = (float*)malloc(nbytes);
    hy = (float*)malloc(nbytes);
    hz = (float*)malloc(nbytes);
    init_array(hx,1.2,N);
    init_array(hy,1.8,N);
    init_array(hz,0,N);

    double start = get_current_time();
    cpu_add(hx,hy,hz,N);
    double end = get_current_time();
    printf("cpu totoal time = %f ms\n",end-start);


    cudaMalloc(&dx,nbytes);
    cudaMalloc(&dy,nbytes);
    cudaMalloc(&dz,nbytes);

    cudaMemcpy(dx,hx,nbytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dy,hy,nbytes,cudaMemcpyHostToDevice);

    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);
    
    float milliseconds=0;
    const int block_size = 512;
    const int grid_size = (N + block_size -1)/ block_size;
    cudaEventRecord(begin);
    cuda_add<<<grid_size,block_size>>>(dx,dy,dz,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, begin, stop);
    printf("gpu totoal time = %f ms\n",milliseconds);

    res = (float*)malloc(nbytes);
    cudaMemcpy(res,dz,nbytes,cudaMemcpyDeviceToHost);


    compare_array(hz,res,N);

    free(hx);
    free(hy);
    free(hz);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    cudaDeviceSynchronize();
    return 0;   
}