#include<stdio.h>
#include<random>
#include<cuda.h>
#include<cuda_runtime.h>





void compare_array(float * x,float *y,int n)
{
    int i=0;
    for(;i<n;i++)
    {
        if(abs(x[i]-y[i])>1e-3)
         {
            //printf("x[%d] = %f,   y[%d] = %f  \n",i,x[i],i,y[i]);
            printf("x and y not equal !\n");
            break;
         }
    }

    //if(i==n)  printf("x = y \n");
}

//input a m*k
//input b k*n
//output c m*n
void sgemm(size_t m,size_t n,size_t k,float* a,float* b,float* c,float* bias)
{
    for(int i=0;i<m;i++)
    {
        float bia = bias[i];   
        for(int j=0;j<n;j++)
        {
            float sum = 0.0;
            for(int z=0;z<k;z++)
            {
                sum += a[i*k+z]*b[z*n+j];
            }

            c[i*n+j] = sum+bia;
        }
    }
}



// template <int BLOCK>
// __global__ void cuda_sgemm_forward(size_t m,size_t n,size_t k,float* a,float* b,float* c,float* bias)
// {
//     int _m = blockIdx.x * BLOCK + threadIdx.x;
//     int _n = blockIdx.y * BLOCK + threadIdx.y;
//     if (_m < m and _n < n) {
//       float sum = 0.f;
//       for (int i = 0; i < k; ++i) {
//         sum += a[_m * k + i] * b[i * n + _n];
//       }
//       c[_m * n + _n] = sum+bias[_m];
//     }
// }


template <int BLOCK>
__global__ void cuda_sgemm_forward(size_t m,size_t n,size_t k,float* a,float* b,float* c,float *bias)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    int global_idx= bx*BLOCK+tx;
    int global_idy= by*BLOCK+ty;

    float *begin_a = a + by * BLOCK * k;
    float *begin_b = b + bx * BLOCK;
    float *end_a = begin_a + k;
    
    int num=0;
    float sum = 0.f;
    for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
         a_ptr += BLOCK, b_ptr += BLOCK * n) {
            num++;
      __shared__ float ashare[BLOCK][BLOCK];
      __shared__ float bshare[BLOCK][BLOCK];
      //printf("global_idy=%d ,global_idx=%d  \n",global_idy,global_idx);
    //   printf("num= %d \n",num);
      if(num==2)
      {
        if(global_idy<m&&global_idx<n){
            ashare[ty][tx] = a_ptr[ty * k + tx];
            bshare[ty][tx] = b_ptr[ty * n + tx];
            printf("global_idy=%d ,global_idx=%d  \n",global_idy,global_idx);
        }
      }
      __syncthreads();

    //   if(num==1)
    //   {
    //     if(global_idx==0&&global_idy==0){
    //         for(int i=0;i<BLOCK;i++){
    //             for(int j=0;j<BLOCK;j++)
    //             {
    //                 printf("%f ",ashare[i][j]);
    //             }
    //             printf("\n");
    //         }
    //       }
    //   }


    //   if(global_idx==1&&global_idy==1) printf("**************\n");

    //   if(global_idx==1&&global_idy==1){
    //     for(int i=0;i<BLOCK;i++){
    //         for(int j=0;j<BLOCK;j++)
    //         {
    //             printf("%f ",bshare[i][j]);
    //         }
    //         printf("\n");
    //     }
    //   }
    //   if(global_idx==1&&global_idy==1) printf("**************\n");
     

  #pragma unroll
      for (int kk = 0; kk < BLOCK; ++kk) {
        sum += ashare[ty][kk] * bshare[kk][tx];
      }
      __syncthreads();
    }
    if(global_idy<m&&global_idx<n)
    {
        c[global_idy*m+global_idx] = sum+bias[global_idy];
    }
    
}

void cuda_sgemm(size_t m,size_t n,size_t k,float* h_a,float* h_b,float* h_c,float* h_bias)
{
    float *d_a,*d_b,*d_c,*d_bias;
    size_t a_nbytes = m*k*sizeof(float);
    size_t b_nbytes = n*k*sizeof(float);
    size_t c_nbytes = m*n*sizeof(float);
    size_t d_nbytes = m*sizeof(float);

    cudaMalloc(&d_a,a_nbytes);
    cudaMalloc(&d_b,b_nbytes);
    cudaMalloc(&d_c,c_nbytes);
    cudaMalloc(&d_bias,d_nbytes);

    cudaMemcpy(d_a,h_a,a_nbytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,b_nbytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias,h_bias,d_nbytes,cudaMemcpyHostToDevice);

    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);
    
    float milliseconds=0;

    constexpr int BLOCK = 16;
    // subm, subn, subk
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  
    
    cudaEventRecord(begin);
    cuda_sgemm_forward<BLOCK><<<grid,block>>>(m,n,k,d_a,d_b,d_c,d_bias);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, begin, stop);
    printf("m=%ld,n=%ld,k=%ld,  gpu totoal time = %f ms\n",m,k,n,milliseconds);


    cudaMemcpy(h_c,d_c,c_nbytes,cudaMemcpyDeviceToHost);

    // for(int i=0;i<m;i++){
    //     for(int j=0;j<n;j++)
    //     {
    //         printf("%f ",h_c[i*m+j]);
    //     }
    //     printf("\n");
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_bias);
}

//矩阵大小从 2 慢慢变大
void test_time()
{
    std::random_device rd;  
    std::mt19937 gen(rd());  
    std::uniform_int_distribution<> dis(1, 10);  

    

    for(int i=17;i<=17;i=i+1)
    {   
        //printf("start i= %d  \n",i);
        size_t nbytes = i*i*sizeof(float);
        float * h_a,*h_b,*h_c,*h_bias;

        h_a = (float *)malloc(nbytes);
        h_b = (float *)malloc(nbytes);
        h_c = (float *)malloc(nbytes);
        h_bias = (float *)malloc(nbytes);
        
        float *result = (float *)malloc(nbytes);
        //printf("mid i= %d  \n",i);
        for(int j=0;j<i*i;j++)
        {
            h_a[j] = dis(gen);
            //printf("a[%d]=%f ",j,h_a[j]);
            //printf("\n");
            h_b[j] = dis(gen);  
        }
        
        for(int k=0;k<i;k++)
        {
            h_bias[k]=0;
        }

        cuda_sgemm(i,i,i,h_a,h_b,h_c,h_bias);
        
        if(i<=4096) 
        {
            sgemm(i,i,i,h_a,h_b,result,h_bias);
            compare_array(result,h_c,i*i);
        }

        free(h_a);
        free(h_b);
        free(h_c);
    }
}

int main()
{


    test_time();
    //test_time();
    printf("01_gemm_naive  run  !!!\n");
    return 0;
}