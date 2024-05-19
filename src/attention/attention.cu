
#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <cmath>




//一般来说输入的是多头注意力
//n_head*(seq_len)*(dims//n_head)

#define N_HEAD 1
#define N 64
#define D 32
#define NUMS N*D*N_HEAD
#define nbytes NUMS*sizeof(float)


/*
Q K V

Q N*D
K N*D
V N*D

*/

//br是Q   ,  一小块的行数
//bc是K V ， 一小块的行数

// size_qolm = br
// size_kv   = bc

//函数调用关系
//flash_attention<32,32,128><<<n_head,32>>>
template<int size_kv,int size_qolm,int dims>
__global__ void flash_attention_gpu(
    const float* q,const float* k,const float* v,float* o,
    const int n_head,const int n,const int d,float scale,
    float *l,float *m,
    const int br,const int bc,
    const int tr,const int tc
)
{
    printf("1111111111111111");
    int tid=threadIdx.x;
    int block_index=blockIdx.x;
    const int nums=n*d;

    //每个block,独立的计算一个head，对于每一个block，确定一下数据指针的指向
    const float *q_head=q+nums*block_index;
    const float *k_head=k+nums*block_index;
    const float *v_head=v+nums*block_index;
    float *o_head=o+nums*block_index;

    const float *l_head=l+n*block_index;
    const float *m_head=m+n*block_index;


    __shared__ float k_share[size_kv*dims];
    __shared__ float v_share[size_kv*dims];

    __shared__ float q_share[size_qolm*dims];
    __shared__ float o_share[size_qolm*dims];

    //__shared__ float l_share[size_qolm];
    //__shared__ float m_share[size_qolm];

    //__shared__ float s_share[size_kv*size_qolm];
    //Load K 𝑗 and V 𝑗 from HBM to on-chip SRAM.
    /*
        共享内存要多大？ 2*(bc*d) 个 float  bc是行数，d是列数，2的话是K+V
    */
    printf("tr=%d ",tr);
    for(int i=0;i<tr;i++)
    {
        


        for(int x=0;x<dims;x++){
            q_share[tid*dims+x]=q_head[i*size_qolm*dims+tid*dims+x];
            o_share[tid*dims+x]=o_head[i*size_qolm*dims+tid*dims+x];
        }
        __syncthreads();
        
        if(tid==0){
            for(int i=0;i<32;i++){
                for(int j=0;j<64;j++){
                    printf("%f ",q[i*64+j]);
                }
                printf("\n");
            }
        }

        //l_share[tid]=l_head[i*size_qolm+tid];
        //m_share[tid]=m_head[i*size_qolm+tid];
        //__syncthreads();


        float m=-100000.f;
        float d=0.0f;

        for(int j=0;j<tc;j++){


            //加载KV到share memory
            //从k_head和v_head，加载size_kv行数据，每行数据有dims
            //1个warp，32个线程，每个线程读取一行的数据到 share memory
            for(int x=0;x<dims;x++){
                k_share[tid*dims+x]=k_head[j*size_kv*dims+tid*dims+x];
                v_share[tid*dims+x]=v_head[j*size_kv*dims+tid*dims+x];
            }
            __syncthreads();

            

            //计算矩阵乘法s=Q*Kt,size_qolm*dims @ size_kv*dims  =size_qolm*size_kv
            //直接对应行相乘就行，每一个线程计算s_share里的,每一行
            
            //s_share[size_qolm*size_kv];
            for(int x=0;x<size_kv;x++){
                float sum=0.0f;
                for(int y=0;y<dims;y++){
                    sum+=q_share[tid*dims+y]*k_share[x*dims+y];
                }
                
                float m_pre=m;
                float d_pre=d;

                m=fmax(m,sum);
                d=d*exp(m_pre-m)+exp(sum-m);

                for(int x=0;x<dims;x++){
                    v_share[tid*dims+x]=o_share[tid*dims+x]*d_pre*exp(m_pre-m)/d+v_share[tid*dims+x]*exp(sum-m)/d;
                }
                __syncthreads();
                //s_share[tid*size_kv+x]=sum;
            }

            
        }

        for(int x=0;x<dims;x++){
            o_head[i*size_qolm*dims+tid*dims+x]=o_share[tid*dims+x];
        }
    }



}



void soft_max(float* data,int cols,int rows)
{   
    for(int i=0;i<rows;i++){
        float max_num=-1000.0f;
        for(int j=0;j<cols;j++){
            max_num=max(max_num,data[i*cols+j]);
        }

        float sum_num=0.0f;
        for(int j=0;j<cols;j++){
            sum_num += exp(data[i*cols+j]-max_num);
        }

        for(int j=0;j<cols;j++){
            data[i*cols+j]= exp(data[i*cols+j]-max_num)/sum_num;
        }

    }

}


//



//
void attention_cpu(float* q,float* k,float* v,float* o,int n_head,int n,int d,float scale)
{
    //printf("scale is %f \n",scale);
    //第一步，计算Q*K，时间复杂度是d*n*n
    float* temp =(float*)malloc(n*n*sizeof(float));

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            float sum=0.0f;
            for(int x=0;x<d;x++){
                sum+=q[i*d+x]*k[j*d+x]; //因为要✖k的转置，这里直接读取行向量即可sum+=q[i*d+k]*k[j*d+k
            }
            temp[i*n+j]=sum*scale;
        }
    }

    // for(int i=0;i<n;i++){
    //     for(int j=0;j<n;j++){
    //         printf("%f ",temp[i*n+j]);
    //     }
    //     printf("\n");
    // }


    //第二步，对temp的每一行做softmax 时间复杂的n*n
    soft_max(temp,n,n);




    //第三部，计算temp*V，时间复杂度是d*n*n
    for(int i=0;i<n;i++){
        for(int j=0;j<d;j++){
            float sum=0.0f;

            for(int x=0;x<n;x++){
                sum+=temp[i*n+x]*v[x*d+j];
            }
            o[i*d+j]=sum;
        }
    }

    free(temp);

}


void attention_cpu_multi_head(float* q,float* k,float* v,float* o,int n_head,int n,int d,float scale)
{
    for(int i=0;i<n_head;i++){

        float* q_head=q+(n*d)*i;
        float* k_head=k+(n*d)*i;
        float* v_head=v+(n*d)*i;

        float* o_head=o+(n*d)*i;

        attention_cpu(q_head,k_head,v_head,o_head,n_head,n,d,scale);

    }
}

bool check_result(float* result,float* groudtruth)
{
    for(int i=0;i<NUMS;i++){
        //printf("result[%d]=%f,groudtruth[%d]=%f \n",i,result[i],i,groudtruth[i]);
        if(abs(result[i]-groudtruth[i])>1e-6){
            printf("result[%d]=%f,groudtruth[%d]=%f \n",i,result[i],i,groudtruth[i]);
            return false;
        }
    }

    return true;
}


void test_attention()
{
    float* q=(float*)malloc(nbytes);
    float* k=(float*)malloc(nbytes);
    float* v=(float*)malloc(nbytes);

    float* o=(float*)malloc(nbytes);

    //init array
    for(int i=0;i<NUMS;i++){
        q[i]=k[i]=v[i]=(float)(i%5);
    }

    float scale=(float)(1/sqrt(D));
    attention_cpu_multi_head(q,k,v,o,N_HEAD,N,D,scale);
    




    float* device_q=nullptr;
    float* device_k=nullptr;
    float* device_v=nullptr;
    float* device_o=nullptr;

    float* device_l=nullptr;
    float* device_m=nullptr;

    cudaMalloc((void **)&device_q, nbytes);
    cudaMalloc((void **)&device_k, nbytes);
    cudaMalloc((void **)&device_v, nbytes);
    cudaMalloc((void **)&device_o, nbytes);

    cudaMalloc((void **)&device_l, nbytes);
    cudaMalloc((void **)&device_m, nbytes);


    cudaMemcpy(device_q, q, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, k, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, v, nbytes, cudaMemcpyHostToDevice);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    flash_attention_gpu<32,32,D><<<N_HEAD,32>>>(device_q,device_k,device_v,device_o,N_HEAD,N,D,scale,device_l,device_m,32,32,N/32,N/32);
    cudaError_t error = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);


    float* result=(float*)malloc(nbytes);
    cudaMemcpy(result, device_o,nbytes , cudaMemcpyDeviceToHost);

    // for(int i=0;i<NUMS;i++){
    //     //printf("11111111");
    //     printf("%f ",result[i]);
    // }
    if(!check_result(result,o)) printf("result is fail \n");
    printf("layernorm_simple latency = %f ms\n", milliseconds);

}



int main(){



    test_attention();
    return 0;
}
