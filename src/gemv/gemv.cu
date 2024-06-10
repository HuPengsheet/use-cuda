#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <cmath>



using namespace std;

//一般来说输入的是多头注意力
//n_head*(seq_len)*(dims//n_head)

#define K 4096
#define N 4096
#define NUMS K*N
#define nbytes NUMS*sizeof(float)



//warp reduce
__device__ float reduce_sum(float val){
    for(int mask=32>>1;mask>=1;mask>>=1){
        val+=__shfl_xor_sync(0xffffffff,val,mask);
    }

    return val;
}

template<int blocksize>
__device__ float block_reduce(float val){

    
    int tid=threadIdx.x;
    
    int warp_num=blocksize/32;
    int warp_index=tid/32;
    int warp_thread=tid%(32);

    __shared__ float temp[blocksize/32];

    val = reduce_sum(val);

    

    if(warp_thread==0) temp[warp_index]=val;
    __syncthreads();
    val = (warp_thread<warp_num)? temp[warp_thread]:0.0f;

    

    val=reduce_sum(val);
    
    return val;

}



// v:k mat:k*n
void cpu_gemv(float* vec,float* mat,float* result,int k,int n)
{   

    for(int i=0;i<n;i++){
        float temp=0.0f;
        for(int j=0;j<k;j++){
            temp+=vec[j]*mat[j*k+i];
        }
        result[i]=temp;
    }
}




//cuda_gemv 01  0.261120 ms
__global__ void cuda_gemv(float* vec,float* mat,float* result,int k,int n)
{
    int tid=threadIdx.x;
    int gtid=blockIdx.x*blockDim.x+tid;
    float sum=0.0f;

    if(gtid<N){
        for(int i=0;i<k;i++){
            sum+=vec[i]*mat[i*k+gtid];
        }
    
        result[gtid]=sum;    
    }
}



//layernorm_simple latency = 0.183296 ms
template<int SIZE>
__global__ void cuda_gemv_2(float* vec,float* mat,float* result,int k,int n)
{
    __shared__ float share_data[SIZE];



    int tid=threadIdx.x;
    int gtid=blockIdx.x*blockDim.x+tid;
    float sum=0.0f;

    //printf("gtid is %d ",gtid);

    for(int i=tid;i<SIZE;i+=blockDim.x){
        share_data[i]=vec[i];
    }
    __syncthreads();

    // if(blockIdx.x==0&&tid==0){
    //     for(int i=0;i<SIZE;i++) printf("%f ",share_data[i]);
    // }

    if(gtid<N){
        for(int i=0;i<k;i++){
            sum+=share_data[i]*mat[i*k+gtid];
        }
    
        result[gtid]=sum;    
    }
}


// layernorm_simple latency = 1.113088 ms
template<int blocksize>
__global__ void cuda_gemv_3(float* vec,float* mat,float* result,int k,int n)
{
    int tid=threadIdx.x;
    int block_index=blockIdx.x;


    float val=0.0f;

    for(int i=tid;i<k;i+=blockDim.x){
        val+=mat[i*K+block_index]*vec[i];
    }
    val = block_reduce<blocksize>(val);
    
    if(tid==0) result[block_index]=val;

}

//Max dimension size of a block size (x,y,z): (1024, 1024, 64)
//Total amount of shared memory per block:       49152 bytes 48kb 12k个 float
template<int blocksize,int nums>
__global__ void cuda_gemv_4(float* vec,float* mat,float* result,int k,int n)
{   
    int tid=threadIdx.x;


    
    float data[nums];
    for(int i=0;i<nums;i++) data[i]=0.0f;
    

    for(int i=0;i<k;i++){
        for(int j=0;(tid+j*blocksize)<n&&j<nums;j++){
            data[j]=data[j]+vec[i]*mat[i*n+tid+j*blocksize];
        }
    }

    for(int j=0;(tid+j*blocksize)<n&&j<nums;j++){
        result[tid+j*blocksize]=data[j];
    }

}




bool check_result(float* result,float* groudtruth)
{
    for(int i=0;i<N;i++){
        //printf("result[%d]=%f,groudtruth[%d]=%f \n",i,result[i],i,groudtruth[i]);
        if(abs(result[i]-groudtruth[i])>1e-6){
            printf("result[%d]=%f,groudtruth[%d]=%f \n",i,result[i],i,groudtruth[i]);
            return false;
        }
    }

    return true;
}


void test_gemv()
{

    float* vec=(float*)malloc(K*sizeof(float));
    float* mat=(float*)malloc(K*N*sizeof(float));
    float* res=(float*)malloc(N*sizeof(float));



    //init array
    for(int i=0;i<K;i++){
        vec[i]=(float)(i%5);
    }    
    for(int i=0;i<NUMS;i++){
        mat[i]=(float)(i%5);
    }

    cpu_gemv(vec,mat,res,K,N);
    


    float* device_vec=nullptr;
    float* device_mat=nullptr;
    float* device_res=nullptr;

    cudaMalloc((void **)&device_vec, K*sizeof(float));
    cudaMalloc((void **)&device_mat, K*N*sizeof(float));
    cudaMalloc((void **)&device_res, N*sizeof(float));

    int BlockSize=256;
    int GridSize=(N+BlockSize-1)/BlockSize;

    dim3 Grid(GridSize);
    dim3 Block(BlockSize);

    cudaMemcpy(device_vec, vec, K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mat, mat, K*N*sizeof(float), cudaMemcpyHostToDevice);

    cuda_gemv<<<Grid,Block>>>(device_vec,device_mat,device_res,K,N);
    cuda_gemv<<<Grid,Block>>>(device_vec,device_mat,device_res,K,N);
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cuda_gemv_2<K><<<Grid,Block>>>(device_vec,device_mat,device_res,K,N);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);


    float* result=(float*)malloc(N*sizeof(float));
    cudaMemcpy(result, device_res,N*sizeof(float),cudaMemcpyDeviceToHost);

    // for(int i=0;i<NUMS;i++){
    //     //printf("11111111");
    //     printf("%f ",result[i]);
    // }
    if(!check_result(res,result)) printf("result is fail \n");
    printf("test_gemv latency = %f ms\n", milliseconds);

}

void test_gemv_3()
{

    float* vec=(float*)malloc(K*sizeof(float));
    float* mat=(float*)malloc(K*N*sizeof(float));
    float* res=(float*)malloc(N*sizeof(float));



    //init array
    for(int i=0;i<K;i++){
        vec[i]=(float)(i%5);
    }    
    for(int i=0;i<NUMS;i++){
        mat[i]=(float)(i%5);
    }

    cpu_gemv(vec,mat,res,K,N);
    


    float* device_vec=nullptr;
    float* device_mat=nullptr;
    float* device_res=nullptr;

    cudaMalloc((void **)&device_vec, K*sizeof(float));
    cudaMalloc((void **)&device_mat, K*N*sizeof(float));
    cudaMalloc((void **)&device_res, N*sizeof(float));

    int BlockSize=256;
    int GridSize=N;

    dim3 Grid(GridSize);
    dim3 Block(BlockSize);

    cudaMemcpy(device_vec, vec, K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mat, mat, K*N*sizeof(float), cudaMemcpyHostToDevice);

    cuda_gemv_3<256><<<Grid,Block>>>(device_vec,device_mat,device_res,K,N);
    cuda_gemv_3<256><<<Grid,Block>>>(device_vec,device_mat,device_res,K,N);
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cuda_gemv_3<256><<<Grid,Block>>>(device_vec,device_mat,device_res,K,N);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);


    float* result=(float*)malloc(N*sizeof(float));
    cudaMemcpy(result, device_res,N*sizeof(float),cudaMemcpyDeviceToHost);

    // for(int i=0;i<NUMS;i++){
    //     //printf("11111111");
    //     printf("%f ",result[i]);
    // }
    if(!check_result(res,result)) printf("result is fail \n");
    printf("test_gemv_3 latency = %f ms\n", milliseconds);

}

void test_gemv_4()
{

    float* vec=(float*)malloc(K*sizeof(float));
    float* mat=(float*)malloc(K*N*sizeof(float));
    float* res=(float*)malloc(N*sizeof(float));



    //init array
    for(int i=0;i<K;i++){
        vec[i]=(float)(i%5);
    }    
    for(int i=0;i<NUMS;i++){
        mat[i]=(float)(i%5);
    }

    cpu_gemv(vec,mat,res,K,N);
    


    float* device_vec=nullptr;
    float* device_mat=nullptr;
    float* device_res=nullptr;

    cudaMalloc((void **)&device_vec, K*sizeof(float));
    cudaMalloc((void **)&device_mat, K*N*sizeof(float));
    cudaMalloc((void **)&device_res, N*sizeof(float));

    constexpr int BlockSize=256;

    int GridSize=1;

    //printf("BlockSize_x=%d,BlockSize_y=%d,thread_num=%d \n",BlockSize_x,BlockSize_y,thread_num);

    dim3 Grid(GridSize);
    dim3 Block(BlockSize);
    constexpr int nums=(N+BlockSize-1)/BlockSize;

    cudaMemcpy(device_vec, vec, K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mat, mat, K*N*sizeof(float), cudaMemcpyHostToDevice);

    cuda_gemv_4<BlockSize,nums><<<Grid,Block>>>(device_vec,device_mat,device_res,K,N);
    cuda_gemv_4<BlockSize,nums><<<Grid,Block>>>(device_vec,device_mat,device_res,K,N);
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cuda_gemv_4<BlockSize,nums><<<Grid,Block>>>(device_vec,device_mat,device_res,K,N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);


    float* result=(float*)malloc(N*sizeof(float));
    cudaMemcpy(result, device_res,N*sizeof(float),cudaMemcpyDeviceToHost);

    // for(int i=0;i<NUMS;i++){
    //     //printf("11111111");
    //     printf("%f ",result[i]);
    // }
    if(!check_result(res,result)) printf("result is fail \n");
    printf("cuda_gemv_4 latency = %f ms\n", milliseconds);

}

int main(){


    test_gemv();
    test_gemv_3();
    test_gemv_4();
    return 0;
}


// for(int i=0;i<K;i++){
//     cout<<vec[i]<<" ";
// } 
// cout<<endl;
// for(int i=0;i<K;i++){
//     for(int j=0;j<N;j++) cout<<mat[i*N+j]<<" ";
//     cout<<endl;
// }
    
// for(int i=0;i<K;i++){
//     cout<<res[i]<<" ";
// } 
// cout<<endl;