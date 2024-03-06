#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
#include <math.h>  
#include<random>

/*

*/

#define BLOCKSIZE 256

bool compare_result(float *x,float*y,size_t n)
{
    for(int i=0;i<n;i++){
        if(abs(x[i]-y[i])>1e-3) return false;
    }


    return true;

}

void init_array(float * array,int n)
{
    std::random_device rd;  
    std::mt19937 gen(rd());  
    std::uniform_int_distribution<> dis(1, 10);  

    for(int i=0;i<n;i++)
    {
        array[i] = dis(gen);
    }
}


void cpu_silu(float *input, float *output, size_t n) {  

    for (size_t i = 0; i < n; i++) {  

        float x = input[i];  

        float sigmoid_part = 1.0f / (1.0f + expf(-x));  

        output[i] = x * sigmoid_part;  

    }  

}


//navie
/*
silu_ n=256  gpu totoal time = 0.099328 ms
silu_ n=512  gpu totoal time = 0.006144 ms
silu_ n=1024  gpu totoal time = 0.013312 ms
silu_ n=2048  gpu totoal time = 0.006144 ms
silu_ n=4096  gpu totoal time = 0.005120 ms
silu_ n=8192  gpu totoal time = 0.004096 ms
silu_ n=16384  gpu totoal time = 0.004096 ms
silu_ n=32768  gpu totoal time = 0.013184 ms
silu_ n=65536  gpu totoal time = 0.011968 ms
silu_ n=131072  gpu totoal time = 0.087584 ms
silu_ n=262144  gpu totoal time = 0.060000 ms
silu_ n=524288  gpu totoal time = 0.072992 ms
silu_ n=1048576  gpu totoal time = 0.075456 ms
silu_ n=2097152  gpu totoal time = 0.087936 ms
silu_ n=4194304  gpu totoal time = 0.108320 ms
silu_ n=8388608  gpu totoal time = 0.150272 ms
silu_ n=16777216  gpu totoal time = 0.231872 ms
silu_ n=33554432  gpu totoal time = 0.392768 ms
*/
__global__ void cuda_silu(float *input,float*output,size_t n)
{
    int tid = threadIdx.x;
    int gtid =blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for(int index=gtid;index<n;index+=stride){
        output[index] = input[index] / (1.0f + expf(-input[index])); 
    }

}

/*
silu_vec n=256 ,gpu totoal time = 0.427008 ms
silu_vec n=512 ,gpu totoal time = 0.006048 ms
silu_vec n=1024 ,gpu totoal time = 0.004096 ms
silu_vec n=2048 ,gpu totoal time = 0.006144 ms
silu_vec n=4096 ,gpu totoal time = 0.006176 ms
silu_vec n=8192 ,gpu totoal time = 0.005120 ms
silu_vec n=16384 ,gpu totoal time = 0.004096 ms
silu_vec n=32768 ,gpu totoal time = 0.005696 ms
silu_vec n=65536 ,gpu totoal time = 0.094336 ms
silu_vec n=131072 ,gpu totoal time = 0.087616 ms
silu_vec n=262144 ,gpu totoal time = 0.008512 ms
silu_vec n=524288 ,gpu totoal time = 0.075328 ms
silu_vec n=1048576 ,gpu totoal time = 0.076992 ms
silu_vec n=2097152 ,gpu totoal time = 0.085856 ms
silu_vec n=4194304 ,gpu totoal time = 0.116736 ms
silu_vec n=8388608 ,gpu totoal time = 0.161120 ms
silu_vec n=16777216 ,gpu totoal time = 0.259680 ms
silu_vec n=33554432 ,gpu totoal time = 0.455616 ms
silu_vec n=67108864 ,gpu totoal time = 0.847808 ms
silu_vec n=134217728 ,gpu totoal time = 1.640768 ms
silu_vec n=268435456 ,gpu totoal time = 3.251488 ms
silu_vec n=536870912 ,gpu totoal time = 6.287680 ms
*/
//usr vector
__global__ void cuda_silu_vec(float *input,float*output,size_t n)
{
    int tid = threadIdx.x;
    int gtid =blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for(int index=gtid;index<n/4;index+=stride){
        float4 a = reinterpret_cast<float4*>(input)[index];
        float4 c;
        c.x = a.x / (1.0f + expf(-a.x)); 
        c.y = a.y / (1.0f + expf(-a.y));
        c.z = a.z / (1.0f + expf(-a.z));
        c.w = a.w / (1.0f + expf(-a.w));
        reinterpret_cast<float4*>(output)[index] = c;
    }

}

bool test_result()
{
    for(int n=3;n<256*1024;n*=3)
    {
        int m = (n+3)/4*4;
        float *hx,*hy ,*cpu_res,*dy,*dx;
        
        size_t nbytes = m*sizeof(float);

        hx=(float*)malloc(nbytes);
        hy=(float*)malloc(nbytes);
        cpu_res=(float*)malloc(nbytes);

        init_array(hx,n);

        cudaMalloc(&dx,nbytes);
        cudaMalloc(&dy,nbytes);
        cudaMemset(dx, 0, nbytes);

        cudaMemcpy(dx,hx,nbytes,cudaMemcpyHostToDevice);
        int gridsize = (m+BLOCKSIZE-1)/BLOCKSIZE;

        cudaEvent_t begin, stop;
        cudaEventCreate(&begin);
        cudaEventCreate(&stop);
        cudaEventRecord(begin);
        float milliseconds=0;
        cuda_silu_vec<<<(gridsize+3)/4,BLOCKSIZE>>>(dx,dy,m);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, begin, stop);

        cudaMemcpy(hy,dy,nbytes,cudaMemcpyDeviceToHost);

        cpu_silu(hx,cpu_res,n);
        bool result = compare_result(cpu_res,hy,n);

        if(result){
            printf("gpu totoal time = %f ms\n",milliseconds);
        }
        else{
            printf("n=%d  fuck!!\n",n);
        }
        cudaFree(dx);
        cudaFree(dy);
        free(hx);
        free(hy);
        free(cpu_res);

    }
}

void test_silu()
{
    printf("test_silu\n");
    for(int n=256;n<256*1024*256*10;n*=2)
    {
        float *hx,*hy ,*cpu_res,*dy,*dx;
        
        size_t nbytes = n*sizeof(float);

        hx=(float*)malloc(nbytes);
        hy=(float*)malloc(nbytes);
        cpu_res=(float*)malloc(nbytes);

        init_array(hx,n);

        cudaMalloc(&dx,nbytes);
        cudaMalloc(&dy,nbytes);

        cudaMemcpy(dx,hx,nbytes,cudaMemcpyHostToDevice);
        int gridsize = (n+BLOCKSIZE-1)/BLOCKSIZE;

        cudaEvent_t begin, stop;
        cudaEventCreate(&begin);
        cudaEventCreate(&stop);
        cudaEventRecord(begin);
        float milliseconds=0;
        cuda_silu<<<gridsize,BLOCKSIZE>>>(dx,dy,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, begin, stop);

        cudaMemcpy(hy,dy,nbytes,cudaMemcpyDeviceToHost);

        cpu_silu(hx,cpu_res,n);
        bool result = compare_result(cpu_res,hy,n);

        if(result){
            printf("silu_ n=%d  gpu totoal time = %f ms\n",n,milliseconds);
        }
        else{
            printf("fuck!!\n");
        }
        cudaFree(dx);
        cudaFree(dy);
        free(hx);
        free(hy);
        free(cpu_res);

    }
    
}

void test_silu_vec()
{
    printf("test_silu_vec\n");
    for(int n=256;n<256*1024*256*10;n*=2)
    {
        float *hx,*hy ,*cpu_res,*dy,*dx;
        
        size_t nbytes = n*sizeof(float);

        hx=(float*)malloc(nbytes);
        hy=(float*)malloc(nbytes);
        cpu_res=(float*)malloc(nbytes);

        init_array(hx,n);

        cudaMalloc(&dx,nbytes);
        cudaMalloc(&dy,nbytes);

        cudaMemcpy(dx,hx,nbytes,cudaMemcpyHostToDevice);
        int gridsize = (n+BLOCKSIZE-1)/BLOCKSIZE;

        cudaEvent_t begin, stop;
        cudaEventCreate(&begin);
        cudaEventCreate(&stop);
        cudaEventRecord(begin);
        float milliseconds=0;
        cuda_silu_vec<<<gridsize/4,BLOCKSIZE>>>(dx,dy,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, begin, stop);

        cudaMemcpy(hy,dy,nbytes,cudaMemcpyDeviceToHost);

        cpu_silu(hx,cpu_res,n);
        bool result = compare_result(cpu_res,hy,n);

        if(result){
            printf("silu_vec n=%d ,gpu totoal time = %f ms\n",n,milliseconds);
        }
        else{
            printf("fuck!!\n");
        }
        cudaFree(dx);
        cudaFree(dy);
        free(hx);
        free(hy);
        free(cpu_res);

    }
    

}


int main()
{
    test_result();
    //test_silu_vec();
    //test_silu();
    //test_silu_vec();
    //test_silu();
    
    return 0;   
}