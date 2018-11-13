
// System includes
#include <stdio.h>
#include <math.h>


// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

__global__ void myKernel2(float *x, long n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (long i = tid; i < n; i += blockDim.x * gridDim.x) {
        ;//sqrt(pow(3.14159, 2));
		x[i] = sqrt(pow(3.14159, (double)x[i]));
    }
}



long NC1 = 131072;
int mainM2(int argc, char **argv)
{
	/*
	srand (time(NULL));
	if (checkCmdLineFlag(argc, (const char **)argv, "N"))
		{
			getCmdLineArgumentValue<long>(argc, (const char **)argv, "N", &N);
		}
		
	float *h_myKernel1Data;
	h_myKernel1Data = (float*)malloc(N * sizeof(float));
	for (long i = 0;i < N;i++)
		h_myKernel1Data[i] = rand() * 1000;
	
	float *myKernel1Data;
	cudaMalloc(&myKernel1Data, N * sizeof(float));
    cudaError_t error;
    error = cudaMemcpy(myKernel1Data, h_myKernel1Data, N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	
	dim3 my1Threads(256, 1);
	if (N % my1Threads.x != 0)
	{
		printf("invalid N\n");
		exit(111000);
	}
	dim3 my1Blocks(sqrt(N / my1Threads.x), sqrt(N / my1Threads.x));
	printf("N: %ld, grid(%d,%d), block(%d,%d)\n", N, my1Blocks.x, my1Blocks.y, my1Threads.x, my1Threads.y);
	myKernel1<<<my1Blocks, my1Threads>>>(myKernel1Data, N);
	error = cudaDeviceSynchronize();
	cudaDeviceReset();
	*/
	
	return 0;
}