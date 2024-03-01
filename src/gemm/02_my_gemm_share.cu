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

    // if(i==n)  printf("x = y \n");
}


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





template <int BLOCK>
__global__ void cuda_sgemm_forward(size_t m,size_t n,size_t k,float* a,float* b,float* c,float *bias)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    int global_idx= bx*BLOCK+tx;
    int global_idy= by*BLOCK+ty;

    float *begin_a = a + bx * BLOCK * k;
    float *end_a = begin_a + k;
    float *a_bottom = a+(m-1)*k;
    float *a_block_bottom = begin_a+(BLOCK-1)*k;
    int a_x_gap = (a_bottom>=a_block_bottom) ? BLOCK : (BLOCK-(a_block_bottom-a_bottom)/k);


    float *begin_b = b + by * BLOCK;
    float *end_b = b+(k-1)*n;
    float *b_right = b+n;
    float *b_block_right = begin_b+BLOCK;
    int b_y_gap = (b_right>=b_block_right) ? BLOCK : (BLOCK-(b_block_right-b_right));
        

    float sum = 0.f;
    int flag=1;
    for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;a_ptr += BLOCK, b_ptr += BLOCK * n) 
    {

        __shared__ float ashare[BLOCK][BLOCK];
        __shared__ float bshare[BLOCK][BLOCK];
        __shared__ float bias_share[BLOCK];

        float* a_block_right = a_ptr+BLOCK;
        int a_y_gap = (end_a>=a_block_right) ? BLOCK : (BLOCK-(a_block_right-end_a));

        float* b_block_bottom = b_ptr+(BLOCK-1) * n;
        int b_x_gap = (end_b>=b_block_bottom) ? BLOCK : (BLOCK-(b_block_bottom-end_b)/n);


        if(tx<a_x_gap&&ty<a_y_gap) ashare[tx][ty] = a_ptr[tx * k + ty];
        if(tx<b_x_gap&&ty<b_y_gap) bshare[tx][ty] = b_ptr[tx * n + ty];
        if(tx<a_x_gap) bias_share[tx] = bias[(begin_a-a)/k+tx];
        __syncthreads();



        #pragma unroll
        for (int kk = 0; kk < BLOCK; ++kk) 
        {
            sum += ashare[tx][kk] * bshare[kk][ty];
        }
        if(flag) 
        {
            sum+=bias_share[tx];
            flag=0;
        }
        __syncthreads();

        ashare[tx][ty]=0;
        bshare[tx][ty]=0; 
        __syncthreads();


    }

    if(global_idx<m&&global_idy<n)
    {
        c[global_idx*n+global_idy] = sum+bias[global_idx];
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

    

    for(int i=2;i<=16384;i=i*2)
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
            h_b[j] = dis(gen);  
        }
        
        for(int k=0;k<i;k++)
        {
            h_bias[k]=0;
        }

        cuda_sgemm(i,i,i,h_a,h_b,h_c,h_bias);
        
        // if(i<=4096) 
        // {
        //     sgemm(i,i,i,h_a,h_b,result,h_bias);
        //     compare_array(result,h_c,i*i);
        // }

        free(h_a);
        free(h_b);
        free(h_c);
    }
}

int main()
{


    test_time();
    test_time();
    printf("01_gemm_naive  run  !!!\n");
    return 0;
}