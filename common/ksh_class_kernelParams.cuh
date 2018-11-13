#include <stdio.h>
#include <fstream>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <ksh_class_definitions.cuh>

#define MAX_PARAMS 10

#pragma once
class kernelParams {
	private:
		int n;
		int added;
	public:
		void* params[MAX_PARAMS];
		float *d_f;
		dim3 offset;
		CUDA_CALLABLE_MEMBER kernelParams();
		CUDA_CALLABLE_MEMBER kernelParams(int);
		CUDA_CALLABLE_MEMBER bool addParameter(void*);
		template <class myType> bool addParameter(myType);
		CUDA_CALLABLE_MEMBER bool addParameter(int);
		CUDA_CALLABLE_MEMBER bool addParameter(float);
		CUDA_CALLABLE_MEMBER bool addParameter(char);
		CUDA_CALLABLE_MEMBER bool addParameter(int*);
		CUDA_CALLABLE_MEMBER bool addParameter(float*);
		CUDA_CALLABLE_MEMBER bool addParameter(char*);
		template <class myType> CUDA_CALLABLE_MEMBER myType getParameter(int);
		template <class myType> CUDA_CALLABLE_MEMBER myType getParameter(int,bool);
		template <class myType> __device__ myType getParameterDevice(int);
		CUDA_CALLABLE_MEMBER void* getParameter(int);
		CUDA_CALLABLE_MEMBER float* getParameterFloat(int);
		CUDA_CALLABLE_MEMBER void setOffset(dim3);
		CUDA_CALLABLE_MEMBER int num();
		
};


CUDA_CALLABLE_MEMBER kernelParams::kernelParams(int nParams)
{
	n = nParams;
	added = 0;
	//offset = dim3(0,0,0);
}

CUDA_CALLABLE_MEMBER kernelParams::kernelParams()
{
	n = MAX_PARAMS;
	added = 0;
	//offset = dim3(0,0,0);
}

CUDA_CALLABLE_MEMBER bool kernelParams::addParameter(void* param)
{
	if (added >= n)
		return false;
	params[added] = param;
	added++;
	return true;
}

template <class myType> bool kernelParams::addParameter(myType param)
{
	cudaError_t err = cudaSuccess;
	if (added >= n)
		return false;
	myType* mt;
	err = cudaMalloc((void**)&mt, sizeof(myType));
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device mt (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(mt, &param, sizeof(myType), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Failed to copy to mt (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	params[added] = mt;
	added++;
	return true;
}

CUDA_CALLABLE_MEMBER bool kernelParams::addParameter(int param)
{
	if (added >= n)
		return false;
	params[added] = &param;
	added++;
	return true;
}

CUDA_CALLABLE_MEMBER bool kernelParams::addParameter(float param)
{
	if (added >= n)
		return false;
	params[added] = &param;
	added++;
	return true;
}

CUDA_CALLABLE_MEMBER bool kernelParams::addParameter(char param)
{
	if (added >= n)
		return false;
	params[added] = &param;
	added++;
	return true;
}

CUDA_CALLABLE_MEMBER bool kernelParams::addParameter(int* param)
{
	if (added >= n)
		return false;
	params[added] = param;
	added++;
	return true;
}

CUDA_CALLABLE_MEMBER bool kernelParams::addParameter(float* param)
{
	if (added >= n)
		return false;
	params[added] = param;
	added++;
	return true;
}

CUDA_CALLABLE_MEMBER bool kernelParams::addParameter(char* param)
{
	if (added >= n)
		return false;
	params[added] = param;
	added++;
	return true;
}

template <class myType> CUDA_CALLABLE_MEMBER myType kernelParams::getParameter(int index)
{
	if (index >= added)
		return NULL;
	myType* p = (myType*)(params[index]);
	return *p;
}

template <class myType> CUDA_CALLABLE_MEMBER myType kernelParams::getParameter(int index, bool isPointer)
{
	if (index >= added)
		return NULL;
	if (isPointer)
		return params[index];
	else
	{
		myType* p = (myType*)(params[index]);
		return p[0];
	}
}

CUDA_CALLABLE_MEMBER void* kernelParams::getParameter(int index)
{
	if (index >= added)
		return NULL;
	return params[index];
}


CUDA_CALLABLE_MEMBER int kernelParams::num()
{
	return added;
}