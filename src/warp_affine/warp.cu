
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include<iostream>
using namespace cv;

#define min(a, b)  ((a) < (b) ? (a) : (b))
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

void warp_affine_bilinear( // 声明
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value
);

Mat warpaffine_to_center_align(const Mat& image, const Size& size){  
    /*
       建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
            思路讲解：https://v.douyin.com/NhrNnVm/
            代码讲解: https://v.douyin.com/NhMv4nr/
    */        

    Mat output(size, CV_8UC3);
    uint8_t* psrc_device = nullptr;
    uint8_t* pdst_device = nullptr;
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    checkRuntime(cudaMalloc(&psrc_device, src_size)); // 在GPU上开辟两块空间
    checkRuntime(cudaMalloc(&pdst_device, dst_size));
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice)); // 搬运数据到GPU上
    
    warp_affine_bilinear(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114
    );

    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(output.data, pdst_device, dst_size, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    return output;
}
double get_current_time()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return usec.count() / 1000.0;
}


#define BLOCKSIZE 256
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
        //init_array(hx,1.0,N);
        

        float result;
        
        //printf("cpu result is %f \n",result);

        cudaMalloc(&dx,x_nbytes);
        cudaMalloc(&dy,y_nbytes);
        cudaMemcpy(dx,hx,x_nbytes,cudaMemcpyHostToDevice);
        

        float* res = (float*)malloc(sizeof(float));
        cudaMalloc(&d_result,sizeof(float));

        cudaSetDevice(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        int GridSize = min((N + BLOCKSIZE - 1) / BLOCKSIZE, deviceProp.maxGridSize[0]);

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
int main(){ 
    /*
    若有疑问，可点击抖音短视频辅助讲解(建议1.5倍速观看) 
        https://v.douyin.com/NhMrb2A/
     */
    // int device_count = 1;
    // checkRuntime(cudaGetDeviceCount(&device_count));

    Mat image = imread("/home/hp/code/github/use-cuda/src/warp_affine/1.jpeg");
    test_reduce();
    double start = get_current_time();
    Mat output = warpaffine_to_center_align(image, Size(640, 640));
    double end = get_current_time();

    printf("total_time=%f  \n",end-start);
    imwrite("output.jpg", output);
    printf("Done. save to output.jpg\n");
    return 0;
}