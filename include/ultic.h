#pragma once

#define CHECK_ERROR(call)\
do{\
	int _error = (call);\
	if(_error)\
	{\
		printf("*** Error *** at [%s:%d] error=%d \n", __FILE__, __LINE__, _error);\
	}\
}while(0)

#define CUDA_CHECK_ERROR(call)\
do{\
	cudaError_t _error = (cudaError_t)(call);\
	if(_error != cudaSuccess)\
	{\
		printf("*** CUDA Error *** at [%s:%d] error=%d, reason:%s \n",\
			__FILE__, __LINE__, _error, cudaGetErrorString(_error));\
	}\
}while(0)