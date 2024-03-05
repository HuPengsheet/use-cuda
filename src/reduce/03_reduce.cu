#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>

/*

*/

#define BLOCKSIZE 256

void init_array(float * array,float x,int n)
{
    for(int i=0;i<n;i++)
    {
        array[i] = x;
    }
}

bool compare_res(float x,float y)
{
    return (x==y)? true:false;
}

void cpu_reduce(float *x,float* res, int n)
{
    float sum =0;
    for(int i=0;i<n;i++){
        sum+=x[i];
    }
    *res = sum;
}


__device__ void warp_reduce(volatile float *sdata,int tid)
{
    float x=sdata[tid];
    if(blockDim.x>=64){
        x+=sdata[tid+32];__syncwarp();
        sdata[tid] = x;__syncwarp();
    }
    x+=sdata[tid+16];__syncwarp();
    sdata[tid] = x;__syncwarp();

    x+=sdata[tid+8];__syncwarp();
    sdata[tid] = x;__syncwarp();

    x+=sdata[tid+4];__syncwarp();
    sdata[tid] = x;__syncwarp();

    x+=sdata[tid+2];__syncwarp();
    sdata[tid] = x;__syncwarp();

    x+=sdata[tid+1];__syncwarp();
    sdata[tid] = x;__syncwarp();
    

}


//输入x,里面含有n个元素
//输出res,里面含有gridDim.x个元素
//对每个block里的数进行规约处理
template <int BLOCK>
__global__ void cuda_reduce(float * x,float * res,size_t n)
{
    __shared__ float sdata[BLOCK];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    float sum=0.0;
    for(int index = global_idx;index<n;index+=stride){
        sum +=x[index];
    }

    sdata[tid] = sum;

    __syncthreads();

    #pragma unroll
    for(int s=BLOCK/2;s>32;s>>=1){

        if(tid<s){
            sdata[tid] = sdata[tid]+sdata[tid+s];
        }
        __syncthreads();
    }

    if(tid<32) warp_reduce(sdata,tid);

    if(tid==0) res[blockIdx.x] = sdata[0];

}

void test_reduce()
{
    //对N个数进行cuda_reduce的sum操作
    for(int N=25600;N<=2560000;N=N*5)
    {
        float *hx,*dx,*dy,*d_result;
        size_t x_nbytes = sizeof(float)*N;
        size_t y_nbytes = sizeof(float)*(N+BLOCKSIZE-1)/BLOCKSIZE;



        hx = (float*)malloc(x_nbytes);
        init_array(hx,1.0,N);
        

        float result;
        cpu_reduce(hx,&result,N);
        

        cudaMalloc(&dx,x_nbytes);
        cudaMalloc(&dy,y_nbytes);
        cudaMemcpy(dx,hx,x_nbytes,cudaMemcpyHostToDevice);
        

        float* res = (float*)malloc(sizeof(float));
        cudaMalloc(&d_result,sizeof(float));

        cudaSetDevice(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        int GridSize = std::min((N + BLOCKSIZE - 1) / BLOCKSIZE, deviceProp.maxGridSize[0]);

        float milliseconds = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cuda_reduce<BLOCKSIZE><<<GridSize,BLOCKSIZE>>>(dx,dy,N);
        cuda_reduce<BLOCKSIZE><<<1,BLOCKSIZE>>>(dy,d_result,GridSize);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaMemcpy(res,d_result,sizeof(float),cudaMemcpyDeviceToHost);

        printf("cpu result is %f gpu result = %f  ",result,*res);
        printf("gpu total time = %f ms\n", milliseconds);
    }
}


int main()
{
    test_reduce();
    test_reduce();
    return 0;   
}