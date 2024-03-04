#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

/*
N=1000 ,gpu totoal time = 0.075776 ms
N=10000 ,gpu totoal time = 0.005120 ms
N=100000 ,gpu totoal time = 0.008864 ms
N=1000000 ,gpu totoal time = 0.088672 ms
N=10000000 ,gpu totoal time = 0.211328 ms
N=100000000 ,gpu totoal time = 1.512512 ms
*/

void init_array(float * array,float x,int n)
{
    for(int i=0;i<n;i++)
    {
        array[i] = x;
    }
}

void compare_array(const float *x,const float * y, int n)
{

    for(int i=0;i<n;i++)
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
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(idx<n)
    {
        z[idx] = x[idx]+y[idx];
    }

}

void test_add()
{
    for(int N=1000;N<=100000000;N=N*10)
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
        printf("N=%d ,gpu totoal time = %f ms\n",N,milliseconds);
    
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
    }
}


int main()
{
    test_add();
    return 0;   
}