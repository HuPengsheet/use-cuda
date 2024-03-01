
template <int BLOCK>
__global__ void cuda_sgemm_forward(size_t m,size_t n,size_t k,float* a,float* b,float* c,float* bias)
{
    int _m = blockIdx.x * BLOCK + threadIdx.x;
    int _n = blockIdx.y * BLOCK + threadIdx.y;
    if (_m < m and _n < n) {
      float sum = 0.f;
      for (int i = 0; i < k; ++i) {
        sum += a[_m * k + i] * b[i * n + _n];
      }
      c[_m * n + _n] = sum+bias[_m];
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
    for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;a_ptr += BLOCK, b_ptr += BLOCK * n) 
    {

        __shared__ float ashare[BLOCK][BLOCK];
        __shared__ float bshare[BLOCK][BLOCK];

        float* a_block_right = a_ptr+BLOCK;
        int a_y_gap = (end_a>=a_block_right) ? BLOCK : (BLOCK-(a_block_right-end_a));

        float* b_block_bottom = b_ptr+(BLOCK-1) * n;
        int b_x_gap = (end_b>=b_block_bottom) ? BLOCK : (BLOCK-(b_block_bottom-end_b)/n);


        if(tx<a_x_gap&&ty<a_y_gap) ashare[tx][ty] = a_ptr[tx * k + ty];
        if(tx<b_x_gap&&ty<b_y_gap) bshare[tx][ty] = b_ptr[tx * n + ty];

        __syncthreads();



        #pragma unroll
        for (int kk = 0; kk < BLOCK; ++kk) 
        {
            sum += ashare[tx][kk] * bshare[kk][ty];
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