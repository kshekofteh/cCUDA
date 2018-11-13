#define Kernel_VADD // ok
//#define Kernel_MMUL // ok
#define Kernel_HS // ok
//#define Kernel_TRAN // ok
//#define Kernel_histogram // ok

//#define Kernel_DXTC
//#define Kernel_MMUL_UM
//#define Kernel_VADD_UM
//#define Kernel_RED
//#define Kernel_BLCK
//#define Kernel_CONVSEP
//#define Kernel_BFS
#define Kernel_C1 // ok

//test for Sync


#include <stdio.h>
#include <fstream>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <ksh_runtime.cuh>

#ifdef Kernel_HS
#include <kernel_hotspot_definitions.cuh>
#include <kernel_hotspot.cuh>
#endif
#ifdef Kernel_VADD
#include <vectorAdd.cu>
#endif
#ifdef Kernel_MMUL
#include <matrixMul.cu>
#endif
#ifdef Kernel_C1
#include <myKernel.cu>
#endif
#ifdef Kernel_TRAN
#include <kernel_transpose.cuh>
#endif
#ifdef Kernel_CONVSEP
#include <convolutionSeparable/convolutionSeparable.cuh>
#include <convolutionSeparable/convolutionSeparable_definitions.cuh>
#include <convolutionSeparable/convolutionSeparable.cuh>
#include <convolutionSeparable/convolutionSeparable_definitions.cuh>
#endif
#ifdef Kernel_histogram
#include <histogram/histogram_gold.cpp>
#include <histogram/histogram256.cu>
#include <histogram/histogram_common.h>
#include <histogram/kernel_histogram_definitions.cuh>
#endif

int main(int argc, char **argv) {
	
	
	cudaProfilerStop();

    int devID = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
		printf("device specified by user: %d\n", devID);
	}
    if (checkCmdLineFlag(argc, (const char **)argv, "silent"))
    {
		_silentVerbos = true;
	}
	int SMX_coeff = 0;
	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		SMX_coeff = getCmdLineArgumentInt(argc, (const char **)argv, "SMX_coeff");
	}

	cudaWrapper wrapper(5);
	wrapper.init(devID, SMX_coeff);
	
	
	
/*******************************************************************************************************************************/	
/************************************************************* HS **************************************************************/	
/*******************************************************************************************************************************/	
#ifdef Kernel_HS
	INIT_hotspot();
	kernelParams paramsHot(14);
	paramsHot.addParameter(MatrixPower);
	paramsHot.addParameter(MatrixTemp[src]);
	paramsHot.addParameter(MatrixTemp[dst]);
	paramsHot.addParameter<int>(total_iterations);
	paramsHot.addParameter<int>(grid_cols);
	paramsHot.addParameter<int>(grid_rows);
	paramsHot.addParameter<int>(borderCols);
	paramsHot.addParameter<int>(borderRows);
	paramsHot.addParameter<float>(Cap);
	paramsHot.addParameter<float>(Rx);
	paramsHot.addParameter<float>(Ry);
	paramsHot.addParameter<float>(Rz);
	paramsHot.addParameter<float>(step);
	paramsHot.addParameter<float>(time_elapsed);
	wrapper.addKernel((kernelPtr)calculate_temp, paramsHot, hs_dimGrid, hs_dimBlock, 0, computeBound, 300);
#endif

/*******************************************************************************************************************************/	
/************************************************************* MM **************************************************************/	
/*******************************************************************************************************************************/	
#ifdef Kernel_MMUL

    //printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

        exit(EXIT_SUCCESS);
    }


    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        //printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    dim3 dimsA(1024, 2048, 1);
    dim3 dimsB(1024, 2048, 1);

    // width of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "wA"))
    {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "hA"))
    {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "wB"))
    {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "hB"))
    {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }

    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

	
	
   // Allocate host memory for matrices A and B
    unsigned long size_A = dimsA.x * dimsA.y;
    unsigned long mem_size_A = sizeof(float) * size_A;
    unsigned long size_B = dimsB.x * dimsB.y;
    unsigned long mem_size_B = sizeof(float) * size_B;
    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned long mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	
	printf("memsize: A: %ld bytes, B: %ld bytes, C: %ld bytes\n", mem_size_A, mem_size_B, mem_size_C);
	//return;
	//srun -p mantaro -w atlas --gres=gpu:2   ./test -hA=16384 -wA=16384 -hB=16384 -wB=16384 -intelligent=0 -numElements=1000000



    float *h_A = (float *)malloc(mem_size_A);
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    float *h_C = (float *) malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }
	


    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
    printf("MatrixA(%d,%d), MatrixB(%d,%d), grid(%d, %d), threads(%d, %d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y, dimsB.x / threads.x, dimsA.y / threads.y, block_size, block_size);
	
	cudaMemset(d_C, -1.9, mem_size_C);
	
	kernelParams paramsMM(5);
	paramsMM.addParameter(d_C);
	//printf("d_C: %d\n", d_C);
	paramsMM.addParameter(d_A);
	paramsMM.addParameter(d_B);
	paramsMM.addParameter<int>(dimsA.x);
	paramsMM.addParameter<int>(dimsB.x);
	//printf("getParameter(0): %d\n", paramsMM.getParameter(0));
	//printf("adding matrixMulCUDA ...\n");
	
	wrapper.addKernel((kernelPtr)matrixMulCUDA, paramsMM, grid, threads, 0, computeBound, 100);
#endif

/*******************************************************************************************************************************/	
/************************************************************** C1 *************************************************************/	
/*******************************************************************************************************************************/	
#ifdef Kernel_C1
	srand (time(NULL));
	if (checkCmdLineFlag(argc, (const char **)argv, "C1"))
		{
			getCmdLineArgumentValue<long>(argc, (const char **)argv, "C1", &N);
		}
		
	float *h_myKernel1Data;
	h_myKernel1Data = (float*)malloc(N * sizeof(float));
	for (long i = 0;i < N;i++)
		h_myKernel1Data[i] = rand() * 1000;
	
	float *myKernel1Data;
	cudaMalloc(&myKernel1Data, N * sizeof(float));
    cudaError_t errorC;
    errorC = cudaMemcpy(myKernel1Data, h_myKernel1Data, N * sizeof(float), cudaMemcpyHostToDevice);
    if (errorC != cudaSuccess)
    {
        printf("cudaMemcpy returned error code %d, line(%d)\n", errorC, __LINE__);
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
	kernelParams paramsC1(2);
	paramsC1.addParameter(myKernel1Data);
	paramsC1.addParameter<long>(N); 
	wrapper.addKernel((kernelPtr)myKernel1, paramsC1, my1Blocks, my1Threads, 0, computeBound, 700);
#endif

/*******************************************************************************************************************************/	
/*******************************************************************************************************************************/	
/*******************************************************************************************************************************/	
#ifdef Kernel_histogram
	INIT_histogram();
	kernelParams paramsHist(3);
	paramsHist.addParameter(d_Histogram);
	paramsHist.addParameter((uint *)d_histogramData);
	uint xyz = byteCount / sizeof(uint);
	paramsHist.addParameter<uint>(xyz);
	wrapper.addKernel((kernelPtr)histogram256Kernel, paramsHist, histogramBlocks, histogramThreads, 0, memoryBound, 75);
#endif

/*******************************************************************************************************************************/	
/************************************************************* TRANSPOSE *******************************************************/	
/*******************************************************************************************************************************/	
#ifdef Kernel_TRAN
	INIT_transpose();
	kernelParams paramsTran(3);
	paramsTran.addParameter(d_odata);
	paramsTran.addParameter(d_idata);
	paramsTran.addParameter<int>(trnsp_size);
	wrapper.addKernel((kernelPtr)transposeNaive, paramsTran, trnsp_grid, trnsp_threads, 0, memoryBound, 80);
#endif
	
/*******************************************************************************************************************************/	
/********************************************************** VECTOR ADD *********************************************************/	
/*******************************************************************************************************************************/	
#ifdef Kernel_VADD
	kernelParams paramsV = prepareVectorAddParams(wrapper, argc, argv); 
#endif
 
/*******************************************************************************************************************************/	
/*************************************************************** CONV **********************************************************/	
/*******************************************************************************************************************************/	
#ifdef Kernel_CONVSEP
	INIT_convolutionSpearable();
	kernelParams paramsConv(5);
	paramsConv.addParameter(d_Buffer);
	paramsConv.addParameter(d_Input);
	paramsConv.addParameter<int>(imageW);
	paramsConv.addParameter<int>(imageH);
	paramsConv.addParameter<int>(imageW);
	wrapper.addKernel((kernelPtr)convolutionRowsKernel, paramsConv, convBlocks, convThreads, 0, memoryBound, 90);
#endif

	if (checkCmdLineFlag(argc, (const char **)argv, "concurrent"))
	{
		//wrapper.classifyKernelsTemp(argc, argv);
		//wrapper.classifyKernels();
		//printf("after wrapper.classifyKernels()\n");
		
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);
		cudaProfilerStart();
		wrapper.launchConcurrent(); 
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaProfilerStop();
		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		printf(COLOR_YELLOW);
		printf("msec: %f\n", msecTotal);
		printf(COLOR_RESET);
		printf("after wrapper.launchConcurrent()\n");
	}
	else if (checkCmdLineFlag(argc, (const char **)argv, "simple"))
	{
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);
		cudaProfilerStart();
		wrapper.launchConcurrentSimple();
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaProfilerStop();
		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		printf(COLOR_YELLOW);
		printf("msec: %f\n", msecTotal);
		printf(COLOR_RESET);
		printf("after wrapper.launchConcurrentSimple()\n");
	}
	else if (checkCmdLineFlag(argc, (const char **)argv, "memcomp"))
	{
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);
		cudaProfilerStart();
		wrapper.launchConcurrentMemoryComputeNoSlice(); 
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaProfilerStop();
		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		printf(COLOR_YELLOW);
		printf("msec: %f\n", msecTotal);
		printf(COLOR_RESET);
		printf("after wrapper.launchConcurrentMemoryComputeNoSlice()\n");
	}
	else if (checkCmdLineFlag(argc, (const char **)argv, "random"))
	{
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);
		cudaProfilerStart();
		wrapper.launchRandom();
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaProfilerStop();
		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		printf(COLOR_YELLOW);
		printf("msec: %f\n", msecTotal);
		printf(COLOR_RESET);
		printf("after wrapper.launchRandom()\n");
	}
	else
	{
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);
		cudaProfilerStart();
		wrapper.launchSerial();
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaProfilerStop();
		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		printf(COLOR_YELLOW);
		printf("msec: %f\n", msecTotal);
		printf(COLOR_RESET);
		printf("after wrapper.launchSerial()\n");
	}
	
	cudaDeviceReset();
	return 0;




// srun -p mantaro -w atlas --gres=gpu:2  nvprof -o test42Int.nvvp -f  ./test -hA=256 -wA=256 -hB=256 -wB=32768 -intelligent=1 -numElements=32768
// srun -p mantaro -w atlas --gres=gpu:2  nvprof -o test42launch.nvvp -f  ./test -hA=256 -wA=256 -hB=256 -wB=32768 -numElements=32768
// srun -p mantaro -w atlas --gres=gpu:2  nvprof -o test42Ser.nvvp -f  ./test -hA=256 -wA=256 -hB=256 -wB=32768 -intelligent=1 -serial=1 -numElements=32768



	
	/*****************************************************************    launch   **********************************************************************************/
	if (checkCmdLineFlag(argc, (const char **)argv, "serial"))
		wrapper.launchSerial();
	else
	{
		if (checkCmdLineFlag(argc, (const char **)argv, "intelligent"))
		{
			if (getCmdLineArgumentInt(argc, (const char **)argv, "intelligent") == 1)
				wrapper.launchConcurrent();
			else
				wrapper.launch();
		}
		else
			wrapper.launch();
	}
	/*****************************************************************    launch   **********************************************************************************/

	cudaDeviceSynchronize();
	cudaDeviceReset();




#ifdef Kernel_MM
    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }
	d_C = (float*)(paramsMM.getParameter(0));
	printf("posProcess d_C: %d\n", d_C);
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("posProcess: cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("Checking computed result for correctness: \n");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6 ; // machine zero

	int errors = 0;
    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/dot_length ;

        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
            correct = false;
			errors++;
        }
		if (errors > 20)
		{
			printf("more than 20 errors.\n");
			break;
		}
    }
	if (correct)
		printf("MM PASS !\n");

#endif

	
	//postProcess(paramsMM);
/*
    err = cudaMemcpy(vecAdd_h_C, vecAdd_d_C, vecAdd_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(vecAdd_h_A[i] + vecAdd_h_B[i] - vecAdd_h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    err = cudaFree(vecAdd_d_A);
    err = cudaFree(vecAdd_d_B);
    err = cudaFree(vecAdd_d_C);
    free(vecAdd_h_A);
    free(vecAdd_h_B);
    free(vecAdd_h_C);
*/
    cudaDeviceReset();


	return 0;
	

}
