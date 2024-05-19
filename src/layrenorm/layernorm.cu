
#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <cmath>

#define ROWS 2048*32
#define COLS 64
#define N ROWS*COLS



__device__ void cuWelfordOnlineSum(
  const float curr,
  float& mu,
  float& sigma2,
  int& count)
{
  count = count + int(1); // 每次调用这个函数，就把处理的数据数量加一。
  float delta = curr - mu; // 看看新数据和现有平均值差多少。
  float lmean = mu + delta / count; // 用这个差值和数据总量来算一个新的平均值。
  mu = lmean; // 把这个新算的平均值记下来。
  float delta2 = curr - lmean; // 现在再算一下新数据和新平均值的差。
  sigma2 = sigma2 + delta * delta2; // 利用这个新旧平均值的差来更新方差。
}



__device__ void WelfordCombine(float b_mean, float b_m2, int b_count, float& mean, float& m2, int& count) {
  if (b_count == 0) { return; }
  float new_count = count + b_count;
  float nb_over_n = b_count/new_count;
  float delta = b_mean - mean;
  mean += delta * nb_over_n;
  m2 += b_m2 + delta * delta * (count) * nb_over_n;
  count = new_count;
}





template<int thread_group_width>
__device__ void WelfordWarpReduce(float& mean,float& m2, int& count) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    float b_mean = __shfl_down_sync(0xffffffff, mean, mask);
    float b_m2 = __shfl_down_sync(0xffffffff, m2, mask);
    int b_count = __shfl_down_sync(0xffffffff, count, mask);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template<int thread_group_width>
 __device__ void WelfordWarpAllReduce(float& mean,float& m2, int& count) {
  WelfordWarpReduce<thread_group_width>(mean, m2, count);
  mean = __shfl_sync(0xffffffff, mean, 0, thread_group_width);
  m2 = __shfl_sync(0xffffffff, m2, 0, thread_group_width);
  count = __shfl_sync(0xffffffff, count, 0, thread_group_width);
}


template<int cols>
__global__ void layernorm(float* data_i,float* data_o)
{
    int tid=threadIdx.x;
    int gtid=blockIdx.x*cols+tid;

    int nums=cols/32;

    float r_data[cols/32];

    int count=0;
    float sigma2b=0.0f;
    float mu=0.0f;

    for(int i=0;i<nums;i++){
        r_data[i]=data_i[gtid+i*32];
        cuWelfordOnlineSum(r_data[i],mu,sigma2b,count);
    }
    //if(blockIdx.x==0)printf("m=%f,  sigma2b=%f,  count=%d \n",mu,sigma2b,count);
    WelfordWarpAllReduce<32>(mu,sigma2b,count);
    sigma2b=sqrt(sigma2b/count);
    
    for(int i=0;i<nums;i++){
        data_o[gtid+i*32]=(r_data[i]-mu)/sigma2b;
    }

    //if(blockIdx.x==0)printf("m=%f,  sigma2b=%f,  count=%d \n",mu,sigma2b,count);

}


__device__ float reduce_sum(float val)
{
    for(int mask=16;mask>=1;mask>>=1){
        val+=__shfl_xor_sync(0xffffffff,val,mask);
    }

    return val;
}


template<int cols>
__global__ void layernorm_simple(float* data_i,float* data_o)
{
    int tid=threadIdx.x;
    int gtid=blockIdx.x*cols+tid;

    int nums=cols/32;

    float r_data[cols/32];


    float m=0.0f;
    float var=0.0f;


    for(int i=0;i<nums;i++){
        r_data[i]=data_i[gtid+i*32];
        m+=r_data[i];
    }

    m=reduce_sum(m);
    m=m/cols;

    for(int i=0;i<nums;i++){
        var+=(r_data[i]-m)*(r_data[i]-m);
    }

    var=reduce_sum(var);
    var=sqrt(var/cols);

    for(int i=0;i<nums;i++){
        data_o[gtid+i*32]=(r_data[i]-m)/var;
    }

}

void cpu_layernorm(float* a,float* b)
{

    for(int i=0;i<ROWS;i++){
        
        float* data_ptr=a+COLS*i;
        float* out=b+COLS*i;

        float e=0.0f;
        float var=0.0f;
        for(int j=0;j<COLS;j++){
            e+=data_ptr[j];
        }

        e=e/COLS;

        for(int j=0;j<COLS;j++){
            var+=(data_ptr[j]-e)*(data_ptr[j]-e);
        }

        var=var/COLS;

        var=sqrt(var);

        for(int j=0;j<COLS;j++){
            out[j]=(data_ptr[j]-e)/var;
        }
        
    }

    //for(int i=0;i<1024;i++) printf("%f ",b[i]);

}

bool check_result(float * result,float *groudtruth)
{
    for(int i=0;i<N;i++){
        if(abs(result[i]-groudtruth[i])>1e-5){
            printf("i==%d,result=%f,groudtruth=%f\n",i,result[i],groudtruth[i]);
            return false;
        }
    }

    return true;
}



void layernorm_test()
{
    int nbytes=sizeof(float)*N;

    float* host_i=(float*)malloc(nbytes);
    float* host_o=(float*)malloc(nbytes);

    for(int i=0;i<N;i++){
        host_i[i]=(float)(i%10);
    }

    cpu_layernorm(host_i,host_o);



    float* device_i=nullptr;
    float* device_o=nullptr;
    cudaMalloc((void **)&device_i, nbytes);
    cudaMalloc((void **)&device_o, nbytes);
    cudaMemcpy(device_i, host_i, nbytes, cudaMemcpyHostToDevice);

    dim3 Grid(ROWS);
    dim3 Block(32);

    layernorm<COLS><<<Grid, Block>>>(device_i, device_o);
    layernorm<COLS><<<Grid, Block>>>(device_i, device_o);
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    layernorm<COLS><<<Grid, Block>>>(device_i, device_o);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);


    float* res=(float*)malloc(nbytes);
    cudaMemcpy(res, device_o,nbytes , cudaMemcpyDeviceToHost);

    if(!check_result(res,host_o)) printf("result is fail \n");
    printf("layernorm latency = %f ms\n", milliseconds);
}


void layernorm_simple_test()
{
    int nbytes=sizeof(float)*N;

    float* host_i=(float*)malloc(nbytes);
    float* host_o=(float*)malloc(nbytes);

    for(int i=0;i<N;i++){
        host_i[i]=(float)(i%10);
    }

    cpu_layernorm(host_i,host_o);



    float* device_i=nullptr;
    float* device_o=nullptr;
    cudaMalloc((void **)&device_i, nbytes);
    cudaMalloc((void **)&device_o, nbytes);
    cudaMemcpy(device_i, host_i, nbytes, cudaMemcpyHostToDevice);

    dim3 Grid(ROWS);
    dim3 Block(32);
    layernorm_simple<COLS><<<Grid, Block>>>(device_i, device_o);
    layernorm_simple<COLS><<<Grid, Block>>>(device_i, device_o);
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    layernorm_simple<COLS><<<Grid, Block>>>(device_i, device_o);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);


    float* res=(float*)malloc(nbytes);
    cudaMemcpy(res, device_o,nbytes , cudaMemcpyDeviceToHost);

    if(!check_result(res,host_o)) printf("result is fail \n");
    printf("layernorm_simple latency = %f ms\n", milliseconds);
}


int main(){
    layernorm_simple_test();
    layernorm_test();
    
}