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


__global__ void cuda_reduce(float * x,float * res,size_t n)
{
    int tid = threadIdx.x;
    int global_idx = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for(int index = global_idx;index<n;index=index+stride)
    {
        for(int s=1;s<BLOCKSIZE;s=s*2)
        {
            if(tid%(s*2)==0)
            {
                 x[index] = x[index]+x[index+s];
            }
            __syncthreads();
        }
        if(tid==0) res[index/stride*gridDim.x+blockIdx.x] = x[index];
    }  
}

void test_reduce()
{
    //对N个数进行cuda_reduce的sum操作
    for(int N=25600;N<=2560000;N=N*10)
    {
        float *hx,*hy,*dx,*dy,*d_result;
        size_t x_nbytes = sizeof(float)*N;
        size_t y_nbytes = sizeof(float)*(N+BLOCKSIZE-1)/BLOCKSIZE;



        hx = (float*)malloc(x_nbytes);
        hy = (float*)malloc(y_nbytes);
        init_array(hx,1.0,N);
        

        float result;
        cpu_reduce(hx,&result,N);
        //printf("cpu result is %f \n",result);

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
        cuda_reduce<<<GridSize,BLOCKSIZE>>>(dx,dy,N);
        cuda_reduce<<<1,BLOCKSIZE>>>(dy,d_result,GridSize);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaMemcpy(res,d_result,x_nbytes,cudaMemcpyDeviceToHost);

        //printf("gpu result = %f \n",res);
        printf("gpu total time = %f ms\n", milliseconds);
    }
}

void test_reduce_2()
{
    int N=25600;
    int gridsize =(N + BLOCKSIZE - 1) / BLOCKSIZE;  //gridsize=1000
    //gridsize=10;

    float * hx,*dx;
    float * hy,*dy;
    int x_nbytes = N*sizeof(float);
    int y_nbytes = gridsize*sizeof(float);

    hx = (float*)malloc(x_nbytes);
    hy = (float*)malloc(y_nbytes);
    init_array(hx,1.0,N);


    cudaMalloc(&dx,x_nbytes);
    cudaMalloc(&dy,y_nbytes);

    cudaMemcpy(dx,hx,x_nbytes,cudaMemcpyHostToDevice);

    float *d_res;
    cudaMalloc(&d_res,sizeof(float));

    float *h_res;
    h_res = (float *)malloc(sizeof(float));
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cuda_reduce<<<gridsize,BLOCKSIZE>>>(dx,dy,N);
    cuda_reduce<<<1,BLOCKSIZE>>>(dy,d_res,gridsize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(hy,dy,y_nbytes,cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_res,d_res,sizeof(float),cudaMemcpyDeviceToHost);
    //printf("res = %f\n",h_res);
    for(int i=0;i<gridsize;i++){
        printf("dy = %f  \n",hy[i]);
    }
    
}

int main()
{
    test_reduce_2();
    return 0;   
}