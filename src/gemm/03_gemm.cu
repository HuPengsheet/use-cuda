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
void sgemm(size_t m,size_t n,size_t k,float* a,float* b,float* c)
{
    for(int i=0;i<m;i++)
    {
        
        for(int j=0;j<n;j++)
        {
            float sum = 0.0;
            for(int z=0;z<k;z++)
            {
                sum += a[i*k+z]*b[z*n+j];
                //printf("%f  %f %d  %f \n",a[i*k+z],b[z*n+j],z*n+j,sum);
            }

            c[i*n+j] = sum;
        }
    }
}



template <int BLOCK,int STRIDE>
__global__ void cuda_sgemm_forward(size_t m,size_t n,size_t k,float* a,float* b,float* c)
{
  // blockIdx control subpanel matrix
  constexpr int STEP = BLOCK * STRIDE;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  float *begin_a = a + by * STEP * k;
  float *begin_b = b + bx * STEP;
  float *end_a = begin_a + k;

  float sum[STRIDE][STRIDE] = {0.f};
  for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += STEP, b_ptr += STEP * n) {
    __shared__ float ashare[STEP][STEP];
    __shared__ float bshare[STEP][STEP];

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        ashare[ty * STRIDE + i][tx * STRIDE + j] =
            a_ptr[(ty * STRIDE + i) * k + tx * STRIDE + j];
        bshare[ty * STRIDE + i][tx * STRIDE + j] =
            b_ptr[(ty * STRIDE + i) * n + tx * STRIDE + j];
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        for (int kk = 0; kk < STEP; ++kk) {
          sum[i][j] +=
              ashare[ty * STRIDE + i][kk] * bshare[kk][tx * STRIDE + j];
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < STRIDE; ++i) {
    for (int j = 0; j < STRIDE; ++j) {
      c[(STEP * by + ty * STRIDE + i) * n + STEP * bx + tx * STRIDE + j] =
          sum[i][j];
    }
  }
}

void cuda_sgemm(size_t m,size_t n,size_t k,float* h_a,float* h_b,float* h_c)
{
    float *d_a,*d_b,*d_c;
    size_t a_nbytes = m*k*sizeof(float);
    size_t b_nbytes = n*k*sizeof(float);
    size_t c_nbytes = m*n*sizeof(float);


    cudaMalloc(&d_a,a_nbytes);
    cudaMalloc(&d_b,b_nbytes);
    cudaMalloc(&d_c,c_nbytes);

    cudaMemcpy(d_a,h_a,a_nbytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,b_nbytes,cudaMemcpyHostToDevice);

    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);
    
    float milliseconds=0;

    constexpr int BLOCK = 16;
    constexpr int STRIDE = 2;
    // subm, subn, subk
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK/STRIDE,(n + BLOCK - 1) / BLOCK/STRIDE);
  
   
    cudaEventRecord(begin);
    cuda_sgemm_forward<BLOCK,STRIDE><<<grid,block>>>(m,n,k,d_a,d_b,d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, begin, stop);
    
    printf("m=%ld,n=%ld,k=%ld,  gpu totoal time = %f ms\n",m,k,n,milliseconds);


    cudaMemcpy(h_c,d_c,c_nbytes,cudaMemcpyDeviceToHost);


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

//æÿ’Û¥Û–°¥” 128 256 512 1024 2048 4096 
void test_time()
{
    std::random_device rd;  
    std::mt19937 gen(rd());  
    std::uniform_int_distribution<> dis(1, 10);  

    

    for(int i=128;i<=16384;i=i*2)
    {   
        //printf("start i= %d  \n",i);
        size_t nbytes = i*i*sizeof(float);
        float * h_a,*h_b,*h_c;

        h_a = (float *)malloc(nbytes);
        h_b = (float *)malloc(nbytes);
        h_c = (float *)malloc(nbytes);
        
        float *result = (float *)malloc(nbytes);
        //printf("mid i= %d  \n",i);
        for(int j=0;j<i*i;j++)
        {
            h_a[j] = dis(gen);
            h_b[j] = dis(gen);  
        }

        cuda_sgemm(i,i,i,h_a,h_b,h_c);
        // sgemm(i,i,i,h_a,h_b,result);
        // compare_array(result,h_c,i*i);
        

        free(h_a);
        free(h_b);
        free(h_c);
    }
}

int main()
{
    // float * out = (float *)malloc(12*sizeof(float));
    // //sgemm(3,4,3,a,b,out);
    
    // cuda_sgemm(3,4,3,a,b,out);
    //compare_array(out,c,12);

    test_time();
    test_time();
    printf("01_gemm_naive  run  !!!\n");
    return 0;
}