/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_functions.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
//vectorAdd2(kernelParams params, dim3 offset, dim3 gridDim)
vectorAdd2(kernelParams params, int offsetX, int offsetY, int offsetZ, dim3 gridDim)
{
	float *A = (float*)(params.getParameter(0));
	float *B = (float*)(params.getParameter(1));
	float *C = (float*)(params.getParameter(2));
	float *temp = (float*)(params.getParameter(3));
	int numElements = params.getParameter<int>(4);
	
	/****************************************************************/
	// rebuild blockId
	dim3 blockIdx = rebuildBlock(offsetX, offsetY, offsetZ);
	/****************************************************************/
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	//int n = numElements;
	
	//if (threadIdx.x == 0)
		//printf("numElements:%d\t\t", numElements);

	//if (threadIdx.x == 0)
		//printf("i:%ld, n:%d [%ld]\t\t\t\t", i, n, n-i);
    if (i < numElements)
    {
		//if (threadIdx.x == 0)
			//printf("ii:%d\t", i);
        C[i] = A[i] + B[i];
 		
		for (int k =0;k < 1;k++)
			for (int j = 2;j < 1000;j++)
			{
				temp[j] = temp[j+1] + temp[j+2];
				//temp[numElements-j] = temp[numElements-j-1] + temp[numElements-j-2];
			}
   }
   //else if (threadIdx.x == 0)
		//printf("iii:%d\t", i);
	   
}

__global__ void
vectorAdd(kernelParams params)
{
	float *A = (float*)(params.getParameter(0));
	float *B = (float*)(params.getParameter(1));
	float *C = (float*)(params.getParameter(2));
	long numElements = params.getParameter<long>(3);
	
	/****************************************************************/
	// rebuild blockId
	dim3 blockIdx = rebuildBlock(params.offset);
	/****************************************************************/
    long i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
 		
   }
}

__global__ void
vectorAddOld(const float *A, const float *B, float *C, long numElements)
{
    long i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
 		
   }
}

kernelParams prepareVectorAddParams(cudaWrapper &wrapper, int argc, char **argv)
{
    cudaError_t err = cudaSuccess;
    long int numElements = 500000;
    if (checkCmdLineFlag(argc, (const char **)argv, "numElements"))
        numElements = getCmdLineArgumentInt(argc, (const char **)argv, "numElements");
    size_t vecAdd_size = numElements * sizeof(float);
    float *vecAdd_h_A = (float *)malloc(vecAdd_size);
    float *vecAdd_h_B = (float *)malloc(vecAdd_size);    // Allocate the host output vector C
    float *vecAdd_h_C = (float *)malloc(vecAdd_size);
    for (int i = 0; i < numElements; ++i)
    {
        vecAdd_h_A[i] = rand()/(float)RAND_MAX;
        vecAdd_h_B[i] = rand()/(float)RAND_MAX;
    }
    float *vecAdd_d_A = NULL;
    err = cudaMalloc((void **)&vecAdd_d_A, vecAdd_size);
    float *vecAdd_d_B = NULL;
    err = cudaMalloc((void **)&vecAdd_d_B, vecAdd_size);
    float *vecAdd_d_C = NULL;
    err = cudaMalloc((void **)&vecAdd_d_C, vecAdd_size);
    err = cudaMemcpy(vecAdd_d_A, vecAdd_h_A, vecAdd_size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(vecAdd_d_B, vecAdd_h_B, vecAdd_size, cudaMemcpyHostToDevice);

    float *d_temp = NULL;
    err = cudaMalloc((void **)&d_temp, vecAdd_size);
	cudaMemset(d_temp, 0, numElements);

    int vecAdd_threadsPerBlock = 128;
	int vecAdd_blocksPerGrid = numElements / vecAdd_threadsPerBlock + (numElements % vecAdd_threadsPerBlock != 0 ? 1 : 0);
	dim3 vecAdd_blocks(vecAdd_blocksPerGrid, 1);
	dim3 vecAdd_threads(vecAdd_threadsPerBlock, 1);
	
	printf("vectorAdd: numElements: %ld, vecAdd_blocks(%d,%d), vecAdd_threads(%d,%d)\n", numElements, vecAdd_blocks.x, vecAdd_blocks.y, vecAdd_threads.x, vecAdd_threads.y);
	
	if (err != cudaSuccess)
		printf("err: %d\n", err);
	
	kernelParams paramsV(5);	
	paramsV.addParameter(vecAdd_d_A);
	paramsV.addParameter(vecAdd_d_B);
	paramsV.addParameter(vecAdd_d_C);
	paramsV.addParameter(d_temp);
	paramsV.addParameter<int>(numElements);
	
	//printf("adding vectorAdd ...\n");
	wrapper.addKernel((kernelPtr)vectorAdd2, paramsV, vecAdd_blocks, vecAdd_threads, 0, memoryBound, 10);
	
	return paramsV;
}

/**
 * Host main routine
 */
 
 /*
int
main2(int argc, char **argv)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its vecAdd_size
    long numElements = 50000;
    if (checkCmdLineFlag(argc, (const char **)argv, "numElements"))
    {
        numElements = getCmdLineArgumentInt(argc, (const char **)argv, "numElements");
    }
    size_t vecAdd_size = numElements * sizeof(float);
    printf("[Vector addition of %ld elements]\n", numElements);

    // Allocate the host input vector A
    float *vecAdd_h_A = (float *)malloc(vecAdd_size);

    // Allocate the host input vector B
    float *vecAdd_h_B = (float *)malloc(vecAdd_size);

    // Allocate the host output vector C
    float *vecAdd_h_C = (float *)malloc(vecAdd_size);

    // Verify that allocations succeeded
    if (vecAdd_h_A == NULL || vecAdd_h_B == NULL || vecAdd_h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        vecAdd_h_A[i] = rand()/(float)RAND_MAX;
        vecAdd_h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *vecAdd_d_A = NULL;
    err = cudaMalloc((void **)&vecAdd_d_A, vecAdd_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *vecAdd_d_B = NULL;
    err = cudaMalloc((void **)&vecAdd_d_B, vecAdd_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *vecAdd_d_C = NULL;
    err = cudaMalloc((void **)&vecAdd_d_C, vecAdd_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(vecAdd_d_A, vecAdd_h_A, vecAdd_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(vecAdd_d_B, vecAdd_h_B, vecAdd_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int vecAdd_threadsPerBlock = 1024;
    //int vecAdd_blocksPerGrid =(numElements + vecAdd_threadsPerBlock - 1) / vecAdd_threadsPerBlock;
	int vecAdd_blocksPerGrid = numElements / vecAdd_threadsPerBlock;
	//vecAdd_blocksPerGrid /= 2;
	dim3 vecAdd_blocks(1, vecAdd_blocksPerGrid, 1);
    //printf("CUDA kernel launch with %d (%d, %d) vecAdd_blocks of %d threads\n", vecAdd_blocks.x, vecAdd_blocks.y, vecAdd_threadsPerBlock);


    cudaError_t error;
    cudaEvent_t vec_start;
    error = cudaEventCreate(&vec_start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create vec_start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t vec_stop;
    error = cudaEventCreate(&vec_stop);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create vec_stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(vec_start, NULL);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record mm_start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    vectorAdd<<<vecAdd_blocks, vecAdd_threadsPerBlock>>>(vecAdd_d_A, vecAdd_d_B, vecAdd_d_C, numElements);

	cudaDeviceSynchronize(); 
    // Record the stop event
    error = cudaEventRecord(vec_stop, NULL);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record vec_stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(vec_stop);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the vec_stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float mmsecTotal = 0.0f;
    error = cudaEventElapsedTime(&mmsecTotal, vec_start, vec_stop);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    printf("Time= %.3f msec\n", mmsecTotal);




    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(vecAdd_h_C, vecAdd_d_C, vecAdd_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(vecAdd_h_A[i] + vecAdd_h_B[i] - vecAdd_h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(vecAdd_d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(vecAdd_d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(vecAdd_d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(vecAdd_h_A);
    free(vecAdd_h_B);
    free(vecAdd_h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

*/