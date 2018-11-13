#include <stdio.h>
#include <fstream>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

#define MAX_PARAMS 10

#define MAX_KERNELS 10

typedef void (*kernelPtrOld)(kernelParams params, dim3 offset, dim3 gridDim);
typedef void (*kernelPtr2)(kernelParams params);
typedef void (*kernelPtr3)(kernelParams params, int x, int y, int z);
typedef void (*kernelPtr)(kernelParams params, int x, int y, int z, dim3 gridDim);

class kernelObj {
	private:
		kernelPtr kernel;
		kernelParams params;
		int sharedMem;
		//cudaStream_t streamID;
	public:
		dim3 gridSize;
		dim3 blockSize;
		kernelClass kClass;
		dim3 notLaunchedGridSize;
		int notLaunchedGridSizeX, notLaunchedGridSizeY, notLaunchedGridSizeZ;
		bool isLeft;
		float CtoM;
		uint64_t sampleKernelTime;
		float sampleKernelTimeFloat;
		int id;
		CUDA_CALLABLE_MEMBER kernelObj();
		CUDA_CALLABLE_MEMBER kernelObj(kernelPtr);
		CUDA_CALLABLE_MEMBER kernelObj(kernelPtr, kernelParams);
		CUDA_CALLABLE_MEMBER kernelObj(kernelPtr, kernelParams, dim3, dim3);
		CUDA_CALLABLE_MEMBER kernelObj(kernelPtr, kernelParams, dim3, dim3, int);
		CUDA_CALLABLE_MEMBER kernelObj(kernelPtr, kernelParams, dim3, dim3, int, kernelClass);
		CUDA_CALLABLE_MEMBER kernelObj(kernelPtr, kernelParams, dim3, dim3, int, kernelClass, int);
		void updateNotLaunchedGridSize(dim3, blockSplitType);
		void launch();
		void launch(dim3, blockSplitType);
		void launch(dim3, dim3, blockSplitType);
		void launch(cudaStream_t);
		void launchLeft(cudaStream_t);
		void launch(dim3, cudaStream_t, blockSplitType);
		void launch(dim3, dim3, cudaStream_t, blockSplitType);
		void launchLeft(dim3, cudaStream_t);
		void setClass(kernelClass);
		void echoRunningInfo();
		//kernelObj(kernelPtr, kernelParams, dim3, dim3, int, int);
		//kernelObj(kernelPtr, kernelParams, dim3, dim3, int, cudaStream_t);
};

CUDA_CALLABLE_MEMBER kernelObj::kernelObj()
{ }

CUDA_CALLABLE_MEMBER kernelObj::kernelObj(kernelPtr kernel) : 
kernel(kernel), params(params), gridSize(gridSize), notLaunchedGridSize(gridSize), blockSize(blockSize), isLeft(true), CtoM(0), sampleKernelTime(0)
{
	notLaunchedGridSizeX = gridSize.x; notLaunchedGridSizeY = gridSize.y; notLaunchedGridSizeZ = gridSize.z;
}

CUDA_CALLABLE_MEMBER kernelObj::kernelObj(kernelPtr kernel, kernelParams params) : 
kernel(kernel), params(params), isLeft(true), CtoM(0), sampleKernelTime(0)
{ }

CUDA_CALLABLE_MEMBER kernelObj::kernelObj(kernelPtr kernel, kernelParams params, dim3 gridSize, dim3 blockSize) : 
kernel(kernel), params(params), gridSize(gridSize), notLaunchedGridSize(gridSize), blockSize(blockSize), isLeft(true), CtoM(0), sampleKernelTime(0)
{ 
	printf("kernelObj kernel:%d, params:%d, grid(%d,%d,%d), block(%d,%d,%d), isLeft:%d\n",
		kernel, params.num(), gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z, isLeft?1:0);

	notLaunchedGridSizeX = gridSize.x; notLaunchedGridSizeY = gridSize.y; notLaunchedGridSizeZ = gridSize.z;
}

CUDA_CALLABLE_MEMBER kernelObj::kernelObj(kernelPtr kernel, kernelParams params, dim3 gridSize, dim3 blockSize, int sharedMem) : 
kernel(kernel), params(params), gridSize(gridSize), notLaunchedGridSize(gridSize), blockSize(blockSize), sharedMem(sharedMem), isLeft(true), CtoM(0), sampleKernelTime(0)
{
	notLaunchedGridSizeX = gridSize.x; notLaunchedGridSizeY = gridSize.y; notLaunchedGridSizeZ = gridSize.z;
}

CUDA_CALLABLE_MEMBER kernelObj::kernelObj(kernelPtr kernel, kernelParams params, dim3 gridSize, dim3 blockSize, int sharedMem, kernelClass kClass) : 
kernel(kernel), params(params), gridSize(gridSize), notLaunchedGridSize(gridSize), blockSize(blockSize), sharedMem(sharedMem), kClass(kClass), isLeft(true), CtoM(0), sampleKernelTime(0)
{ 
	//printf("# a kernel added. type: %d\n", kClass);

	notLaunchedGridSizeX = gridSize.x; notLaunchedGridSizeY = gridSize.y; notLaunchedGridSizeZ = gridSize.z;
}

CUDA_CALLABLE_MEMBER kernelObj::kernelObj(kernelPtr kernel, kernelParams params, dim3 gridSize, dim3 blockSize, int sharedMem, kernelClass kClass, int Id) : 
kernel(kernel), params(params), gridSize(gridSize), notLaunchedGridSize(gridSize), blockSize(blockSize), sharedMem(sharedMem), kClass(kClass), isLeft(true), CtoM(0), sampleKernelTime(0), id(Id)
{
	notLaunchedGridSizeX = gridSize.x; notLaunchedGridSizeY = gridSize.y; notLaunchedGridSizeZ = gridSize.z;
}
/*
kernelObj::kernelObj(kernelPtr kernel, kernelParams params, dim3 gridSize, dim3 blockSize, int sharedMem, int streamID) : 
kernel(kernel), params(params), gridSize(gridSize), blockSize(blockSize), sharedMem(sharedMem), streamID((cudaStream_t)streamID)
{ }

kernelObj::kernelObj(kernelPtr kernel, kernelParams params, dim3 gridSize, dim3 blockSize, int sharedMem, cudaStream_t streamID) : 
kernel(kernel), params(params), gridSize(gridSize), blockSize(blockSize), sharedMem(sharedMem), streamID(streamID)
{ }
*/
void kernelObj::updateNotLaunchedGridSize(dim3 runGridSize, blockSplitType split)
{
	if (notLaunchedGridSize.x == runGridSize.x &&
		notLaunchedGridSize.y == runGridSize.y &&
		notLaunchedGridSize.z == runGridSize.z)
	{
		isLeft = false;
		notLaunchedGridSize.x = 0;
		notLaunchedGridSize.y = 0;
		notLaunchedGridSize.z = 0;
		notLaunchedGridSizeX = 0;
		notLaunchedGridSizeY = 0;
		notLaunchedGridSizeZ = 0;
		return;
	}
	if (notLaunchedGridSizeX == runGridSize.x &&
		notLaunchedGridSizeY == runGridSize.y &&
		notLaunchedGridSizeZ == runGridSize.z)
	{
		isLeft = false;
		notLaunchedGridSize.x = 0;
		notLaunchedGridSize.y = 0;
		notLaunchedGridSize.z = 0;
		notLaunchedGridSizeX = 0;
		notLaunchedGridSizeY = 0;
		notLaunchedGridSizeZ = 0;
		return;
	}
	isLeft = true;
	if ((split & xSplit) != 0)
	{
		notLaunchedGridSize.x = Max(notLaunchedGridSize.x - runGridSize.x, 0);
		notLaunchedGridSizeX = notLaunchedGridSize.x;
		if (!notLaunchedGridSize.x)
			isLeft = false;
	}
	if ((split & ySplit) != 0)
	{
		notLaunchedGridSize.y = Max(notLaunchedGridSize.y - runGridSize.y, 0);
		notLaunchedGridSizeY = notLaunchedGridSize.y;
		if (!notLaunchedGridSize.y)
			isLeft = false;
	}
	if ((split & zSplit) != 0)
	{
		notLaunchedGridSize.z = Max(notLaunchedGridSize.z - runGridSize.z, 0);
		notLaunchedGridSizeZ = notLaunchedGridSize.z;
		if (!notLaunchedGridSize.z)
			isLeft = false;
	}
}
void kernelObj::launch()
{
	printf("launch, gridSize(%d,%d,%d)\n", gridSize.x, gridSize.y, gridSize.z);
	//kernel<<<gridSize, blockSize>>>(params, dim3(0,0,0), gridSize);     ////ZZZ 960911 غیر فعال شد چون خطای اجرا نکردن ضرب ماتریس بزرگ بود
	kernel<<<gridSize, blockSize>>>(params, 0,0,0, gridSize);
}
void kernelObj::launch(dim3 runGridSize, blockSplitType split)
{
	dim3 ln = min(runGridSize, dim3(notLaunchedGridSizeX, notLaunchedGridSizeY, notLaunchedGridSizeZ));
	printf("kernelObj::launch(%d,%d,%d), splitType:%d\n", ln.x, ln.y, ln.z, split);
	//kernel<<<ln, blockSize, sharedMem>>>(params, dim3(0,0,0), gridSize);     ////ZZZ 960911 غیر فعال شد چون خطای اجرا نکردن ضرب ماتریس بزرگ بود
	kernel<<<ln, blockSize, sharedMem>>>(params, 0,0,0, gridSize);
	updateNotLaunchedGridSize(runGridSize, split);
}
void kernelObj::launch(dim3 runGridSize, dim3 offset, blockSplitType split)
{
	params.offset = offset;
	dim3 ln = min(runGridSize, dim3(notLaunchedGridSizeX, notLaunchedGridSizeY, notLaunchedGridSizeZ));
	//kernel<<<ln, blockSize, sharedMem>>>(params, offset, gridSize);     ////ZZZ 960911 غیر فعال شد چون خطای اجرا نکردن ضرب ماتریس بزرگ بود
	kernel<<<ln, blockSize, sharedMem>>>(params, offset.x, offset.y, offset.z, gridSize);
	updateNotLaunchedGridSize(runGridSize, split);
}
void kernelObj::launch(cudaStream_t streamID)
{
	//kernel<<<gridSize, blockSize, sharedMem, streamID>>>(params, dim3(0,0,0), gridSize);         ////ZZZ 960911 غیر فعال شد چون خطای اجرا نکردن ضرب ماتریس بزرگ بود
	kernel<<<gridSize, blockSize, sharedMem, streamID>>>(params, 0,0,0, gridSize);
}
void kernelObj::launchLeft(cudaStream_t streamID)
{
	// kernel<<<min(gridSize,notLaunchedGridSize), blockSize, sharedMem, streamID>>>(params, dim3(0,0,0), gridSize);     // 960911 غیر فعال شد چون خطای اجرا نکردن ضرب ماتریس بزرگ بود
	kernel<<<min(gridSize,notLaunchedGridSize), blockSize, sharedMem, streamID>>>(params, 0,0,0, gridSize);
}
void kernelObj::launch(dim3 runGridSize, cudaStream_t streamID, blockSplitType split)
{
	dim3 GS = min(runGridSize,dim3(notLaunchedGridSizeX, notLaunchedGridSizeY, notLaunchedGridSizeZ));
	//printf("GS(%d,%d,%d), stream:%d\n", GS.x, GS.y, GS.z, streamID);
	//kernel<<<GS, blockSize, sharedMem, streamID>>>(params, dim3(0,0,0), gridSize);     ////ZZZ 960911 غیر فعال شد چون خطای اجرا نکردن ضرب ماتریس بزرگ بود
	kernel<<<GS, blockSize, sharedMem, streamID>>>(params, 0,0,0, gridSize);
	updateNotLaunchedGridSize(runGridSize, split);
}
void kernelObj::launch(dim3 runGridSize, dim3 offset, cudaStream_t streamID, blockSplitType split)
{
	//dim3 offsetTemp = params.offset;
	//params.offset = offset;
	dim3 GS = min(runGridSize,dim3(notLaunchedGridSizeX, notLaunchedGridSizeY, notLaunchedGridSizeZ));
	//printf("GS(%d,%d,%d), offset(%d,%d,%d), stream:%d\n", GS.x, GS.y, GS.z, offset.x, offset.y, offset.z, streamID);
	kernel<<<GS, blockSize, sharedMem, streamID>>>(params, offset.x, offset.y, offset.z, gridSize);
	//params.offset = offsetTemp;
	updateNotLaunchedGridSize(runGridSize, split);
}

void kernelObj::launchLeft(dim3 offset, cudaStream_t streamID)
{
	if (!isLeft)
		return;
	//kernel<<<notLaunchedGridSize, blockSize, sharedMem, streamID>>>(params, offset, gridSize);     ////ZZZ 960911 غیر فعال شد چون خطای اجرا نکردن ضرب ماتریس بزرگ بود
	kernel<<<notLaunchedGridSize, blockSize, sharedMem, streamID>>>(params, offset.x, offset.y, offset.z,  gridSize);
	isLeft = false;
}

void kernelObj::setClass(kernelClass KClass)
{
	kClass = KClass;
}

void kernelObj::echoRunningInfo()
{
	printf("======== running info of kernel: %d ========\n", id);
	printf("gridSize(%d,%d,%d), blockSize(%d,%d,%d)\n", gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);
	printf("notLaunchedGridSize(%d,%d,%d)\n", notLaunchedGridSize.x, notLaunchedGridSize.y, notLaunchedGridSize.z);
	printf("notLaunchedGridSizeX:%d, notLaunchedGridSizeY:%d, notLaunchedGridSizeZ:%d)\n", notLaunchedGridSizeX, notLaunchedGridSizeY, notLaunchedGridSizeZ);
	printf(" \n");
	//printf("========                             ========\n");
}