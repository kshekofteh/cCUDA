#include <stdio.h>
#include <fstream>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include <cupti.h>
#include <ksh_cupti_cuptiObj.cuh>
#include <ksh_smInfo.cuh>
#include <ksh_framework.cuh>
#include <ksh_ioOperation.cuh>
#include <ksh_eventOperation.cuh>
#include <ksh_class_kernelParams.cuh>
#include <ksh_class_kernelObj.cuh>

bool _silentVerbos = false;

class cudaWrapper {
	private:
		kernelObj* kernels;
		int nKernels;
		int nAddedKernels;
		cudaStream_t* streams;
		int _device;
		int _SMXs;
		int _CoresPerSMX;
		float _peakPerformanceSP;
		float _peakPerformanceDP;
		float _peakBandwidth;
		float _intensityThresholdSP;
		float _intensityThresholdDP;
	public:
		cudaWrapper();
		cudaWrapper(int);
		bool init();
		bool init(int,int);
		void addKernel(kernelPtr);
		void addKernel(kernelPtr, kernelParams);
		void addKernel(kernelPtr, kernelParams, dim3, dim3);
		void addKernel(kernelPtr, kernelParams, dim3, dim3, int);
		void addKernel(kernelPtr, kernelParams, dim3, dim3, int, kernelClass);
		void addKernel(kernelPtr, kernelParams, dim3, dim3, int, kernelClass, int);
		//void addKernel(void (*kernelPointer)(dim3 offset, void** params));
		//void addParameters(kernelPtr, kernelParams*);
		//void addExConfig(void* kernelPointer, dim3 gridSize, dim3 blockSize, int sharedMem);
		void launch();
		void launchRandom();
		void launchSerial();
		void launchConcurrentMemoryCompute();
		void launchConcurrentMemoryComputeNoSlice();
		void launchConcurrent();
		void launchConcurrentSimple();
		void classifyKernels();
		void classifyKernelsTemp(int argc, char **argv);
		int blocksToRunCalc(dim3);
		void executionTimeCalc();
		
};

cudaWrapper::cudaWrapper()
{
	nKernels = MAX_KERNELS;
	nAddedKernels = 0;
	kernels = (kernelObj*)malloc(nKernels * sizeof(kernelObj));
	streams = (cudaStream_t*)malloc(nKernels * sizeof(cudaStream_t));
	_device = -1;
	_SMXs = 0;
	_CoresPerSMX = 0;
	_intensityThresholdSP = 0;
	_intensityThresholdDP = 0;
	_peakPerformanceSP = 0;
	_peakPerformanceDP = 0;
	_peakBandwidth = 0;
}

cudaWrapper::cudaWrapper(int numKernels)
{
	nKernels = numKernels;
	nAddedKernels = 0;
	kernels = (kernelObj*)malloc(nKernels * sizeof(kernelObj));
	streams = (cudaStream_t*)malloc(nKernels * sizeof(cudaStream_t));
	_device = -1;
	_SMXs = 0;
	_CoresPerSMX = 0;
	_intensityThresholdSP = 0;
	_intensityThresholdDP = 0;
	_peakPerformanceSP = 0;
	_peakPerformanceDP = 0;
	_peakBandwidth = 0;
}

bool cudaWrapper::init(int device, int SMX_coeff = 1)
{
	_device = device;
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        return false;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
		return false;
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
	cudaSetDevice(_device);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, _device);
	_SMXs = deviceProp.multiProcessorCount * SMX_coeff;
	_CoresPerSMX = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	
	_peakPerformanceSP = 11000;
	_peakPerformanceDP = 11000;
	_peakBandwidth = 480;
	_intensityThresholdSP = _peakPerformanceSP / _peakBandwidth;
	_intensityThresholdDP = _peakPerformanceDP / _peakBandwidth;
	
	printf("name: %s, SMs: %d, CoresPerSM: %d\n==============\n==============\n==============\n", deviceProp.name, _SMXs, _CoresPerSMX);
	return true;
	
}

bool cudaWrapper::init()
{
	return init(0);
}

void cudaWrapper::addKernel(kernelPtr kernelPointer)
{
	kernels[nAddedKernels] = kernelObj(kernelPointer);
	nAddedKernels++;
}

void cudaWrapper::addKernel(kernelPtr kernelPointer, kernelParams params)
{
	kernels[nAddedKernels] = kernelObj(kernelPointer, params);
	nAddedKernels++;
}

void cudaWrapper::addKernel(kernelPtr kernelPointer, kernelParams params, dim3 gridSize, dim3 blockSize)
{
	kernels[nAddedKernels] = kernelObj(kernelPointer, params, gridSize, blockSize);
	nAddedKernels++;
}

void cudaWrapper::addKernel(kernelPtr kernelPointer, kernelParams params, dim3 gridSize, dim3 blockSize, int sharedMem, kernelClass kClass)
{
	kernels[nAddedKernels] = kernelObj(kernelPointer, params, gridSize, blockSize, sharedMem, kClass);
	nAddedKernels++;
}

void cudaWrapper::addKernel(kernelPtr kernelPointer, kernelParams params, dim3 gridSize, dim3 blockSize, int sharedMem, kernelClass kClass, int id)
{
	kernels[nAddedKernels] = kernelObj(kernelPointer, params, gridSize, blockSize, sharedMem, kClass, id);
	nAddedKernels++;
}

void calcSplit(int epochBlock, dim3 gridSize, dim3 &epochGridSize, dim3 &offset, blockSplitType &splitType)
{
	int totalBlocks = gridSize.x * gridSize.y * gridSize.z;
	if (totalBlocks <= epochBlock)
	{
		splitType = noSplit;
		epochGridSize = gridSize;
		offset = dim3(0,0,0);
		return;
	}
	
	int sp = totalBlocks / epochBlock;
	
	splitType = xSplit;
	int parts = (int)ceil((float)gridSize.x / (float)sp);
	epochGridSize = dim3(min(epochBlock, max(parts, 1)), gridSize.y, gridSize.z);
	//epochGridSize = dim3(min(epochBlock, max(parts, 1)), gridSize.y, gridSize.z);
	offset = dim3(min(epochBlock, max(parts, 1)), 0, 0);
	if (gridSize.x < gridSize.y)
	{
		int parts = (int)ceil((float)gridSize.y / (float)sp);
		splitType = ySplit;
		epochGridSize = dim3(gridSize.x, min(epochBlock, max(parts, 1)), gridSize.z);
		offset = dim3(0, min(epochBlock, max(parts, 1)), 0);
		if (gridSize.y < gridSize.z)
		{
			int parts = (int)ceil((float)gridSize.z / (float)sp);
			splitType = zSplit;
			epochGridSize = dim3(gridSize.x, gridSize.y, min(epochBlock, max(parts, 1)));
			offset = dim3(0, 0, min(epochBlock, max(parts, 1)));
		}
	}
	else if (gridSize.x < gridSize.z)
	{
		splitType = zSplit;
		int parts = (int)ceil((float)gridSize.z / (float)sp);
		epochGridSize = dim3(gridSize.x, gridSize.y, min(epochBlock, max(parts, 1)));
		offset = dim3(0, 0, min(epochBlock, max(parts, 1)));
	}
}

int cudaWrapper::blocksToRunCalc(dim3 BlockSize)
{
	int threadsInBlock = BlockSize.x * BlockSize.y * BlockSize.z;
	int blocksToRun = _SMXs;
	if (threadsInBlock < _CoresPerSMX)
	{
		printf("less threads than cores in a SMX!\n");
		blocksToRun *= (  (_CoresPerSMX / threadsInBlock) + (_CoresPerSMX % threadsInBlock ? 1 : 0)  );
	}
	return blocksToRun;
}

void cudaWrapper::executionTimeCalc()
{
	if (!_silentVerbos)
		printf("\n=========================\nStart of executionTimeCalc()\n\n");
	int nKernel = min(nKernels, nAddedKernels);
	blockSplitType splitType;
	dim3 epochGridSize, offset;
	for (int i = 0;i < nKernel;i++)
	{
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);
		int blocksToRun = blocksToRunCalc(kernels[i].blockSize);
		calcSplit(blocksToRun, kernels[i].gridSize, epochGridSize, offset, splitType);
		kernels[i].launch(epochGridSize, splitType);
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		//cudaDeviceSynchronize();
		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		kernels[i].sampleKernelTimeFloat = msecTotal;
		if (!_silentVerbos)
			printf(COLOR_MAGENTA);
		if (!_silentVerbos)
			printf("kernel.id:%d, msec:%f\n", kernels[i].id, msecTotal);
		if (!_silentVerbos)
			printf(COLOR_RESET);
	}
	if (!_silentVerbos)
		printf("\nEnd of executionTimeCalc()\n=========================\n\n");
}

void cudaWrapper::classifyKernels()
{
	cudaProfilerStop();
	
	
	const int metricInstCount = 10;
	const char* metrics_inst[metricInstCount] = { 
		"flop_count_hp",
		"flop_count_sp",
		"flop_count_dp",
		"flop_count_sp_special",
		"inst_integer",
		"inst_bit_convert",
		"inst_control",
		"inst_compute_ld_st",
		"inst_misc",
		"inst_inter_thread_communication",
	};
	const int metricMemCount = 2;
	const char* metrics_mem[metricMemCount] = { 
		//"shared_store_transactions", // not used
		//"shared_load_transactions", // not used
		"gld_transactions", 
		"gst_transactions", 
		//"dram_read_transactions",
		//"dram_write_transactions",
		//"sysmem_read_transactions",
		//"sysmem_write_transactions",
		};
	const int metricOtherCount = 2;
	const char* metrics_other[metricOtherCount] = { 
		"inst_executed",
		"inst_issued",
	};
	bool read_otherMetrics = false;
		
	if (!_silentVerbos)
		printf("at least %d blocks for each metric(%d), each block at least %d threads for each kernel should be launched for profiling...\n", _SMXs, metricInstCount+metricMemCount, _CoresPerSMX);
	
	executionTimeCalc();
	
	cuptiObj cuptiOb;
	cuptiOb.init();
	cuptiOb.registerCallbacks();
		
	int nKernel = min(nKernels, nAddedKernels);
	for (int i = 0;i < nKernel;i++)
	{
		if (!_silentVerbos)
			printf("================\nkernel %d Grid:(%d,%d,%d), Block:(%d,%d,%d) \n", i, 
			kernels[i].gridSize.x, kernels[i].gridSize.y, kernels[i].gridSize.z,
			kernels[i].blockSize.x, kernels[i].blockSize.y, kernels[i].blockSize.z);
		blockSplitType splitType;
		dim3 epochGridSize;
		dim3 offset;
		
		int blocksToRun = blocksToRunCalc(kernels[i].blockSize);
		calcSplit(blocksToRun, kernels[i].gridSize, epochGridSize, offset, splitType);
		
		kernels[i].launch(epochGridSize, splitType);
		cudaDeviceSynchronize();
		cuptiOb.flushAll();
		uint64_t kernelDuration = cuptiOb.kernelDuration;
		kernels[i].sampleKernelTime = cuptiOb.kernelDuration;
		
		if (!_silentVerbos)
			printf("\tfirst run (%d,%d,%d): kernelDuration:%d \n", epochGridSize.x, epochGridSize.y, epochGridSize.z, kernelDuration);
		fflush(stdout);
		
		
		
		////////////////////////// test whether really there is a need to kernelDuration or not???
		
		

/*
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);
		kernels[i].launch(epochGridSize, splitType);
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		kernels[i].sampleKernelTimeFloat = msecTotal;
		if (!_silentVerbos)
			printf("msec: %f\n", msecTotal);
*/		
		//kernelDuration
		 
		// use cudaEvents instead !!!!!!!!!!!!
		
		int pd = 1;
		unsigned long long n_insts = 0;
		unsigned long long n_mems = 0;

		CUpti_MetricValue values_inst[metricInstCount];
		if (!_silentVerbos)
			printf("\treading %d inst metrics\n", metricInstCount);
		for (int j = 0;j < metricInstCount;j++)
		{
			//printf("\nto setup ...");
			if (cuptiOb.setup(metrics_inst[j]) != 0)
			{
				printf("\n\t\tinvalid metric: %s\n", metrics_inst[j]);
				continue;
			}
			else
			{
				int numPass = cuptiOb.getNumPass();
				if (numPass > 0)
				{
					//printf("\tmetric[%d]: %s , numPass: %d\n", j, metrics_inst[j], numPass);
					for (int pass = 0; pass < numPass; pass++) {
						cuptiOb.startNewPass(pass);
						dim3 offset = multiply(epochGridSize, pd++, splitType);
						kernels[i].launch(epochGridSize, offset, splitType);
						if (!kernels[i].isLeft)
							break;
					}
					int stat = cuptiOb.checkAfterPass();
					values_inst[j] = cuptiOb.metricGenerateValue();
					n_insts += (unsigned long long)values_inst[j].metricValueUint64;
					//printf("\t\tmetric[%d] %s: %llu\n", j, metrics_inst[j], (unsigned long long)values_inst[j].metricValueUint64);
				}
				else
				{
					printf("\t\tmetric[%d] %s: !!! numPass=0\n", j, metrics_inst[j]);
				}
				fflush(stdout);
				
				cuptiOb.unsetup();
			}
			///////////////// test it
		}

		CUpti_MetricValue values_mem[metricMemCount];
		if (!_silentVerbos)
			printf("\treading %d memory metrics\n", metricMemCount);
		for (int j = 0;j < metricMemCount;j++)
		{
			//printf("\nto setup ...");
			if (cuptiOb.setup(metrics_mem[j]) != 0)
			{
				printf("\n\t\tinvalid metric: %s\n", metrics_mem[j]);
				continue;
			}
			else
			{
				int numPass = cuptiOb.getNumPass();
				if (numPass > 0)
				{
					//printf("\tmetric[%d]: %s , numPass: %d\n", j, metrics_mem[j], numPass);
					for (int pass = 0; pass < numPass; pass++) {
						cuptiOb.startNewPass(pass);
						dim3 offset = multiply(epochGridSize, pd++, splitType);
						kernels[i].launch(epochGridSize, offset, splitType);
						if (!kernels[i].isLeft)
							break;
					}
					int stat = cuptiOb.checkAfterPass();
					values_mem[j] = cuptiOb.metricGenerateValue();
					n_mems += (unsigned long long)values_mem[j].metricValueUint64;
					//printf("\t\tmetric[%d] %s: %llu\n", j, metrics_mem[j], (unsigned long long)values_mem[j].metricValueUint64);
				}
				else
				{
					printf("\t\tmetric[%d] %s: !!! numPass=0\n", j, metrics_mem[j]);
				}
				fflush(stdout);
				
				cuptiOb.unsetup();
			}
			///////////////// test it
		}
		
		//double gflops = ((double)n_insts/1000000)/(kernelDuration/1000);
		double ratio = (double)n_insts/(double)n_mems;
		if (!_silentVerbos)
			printf("\tn_insts:%llu, n_mems:%llu, ratio:%g\n", n_insts, n_mems, ratio);
		kernels[i].CtoM = ratio;
		

		if (read_otherMetrics)
		{
			CUpti_MetricValue values_other[metricOtherCount];
			if (!_silentVerbos)
				printf("\treading %d other metrics\n", metricOtherCount);
			for (int j = 0;j < metricOtherCount;j++)
			{
			//printf("\nto setup ...");
			if (cuptiOb.setup(metrics_other[j]) != 0)
			{
				printf("\n\t\tinvalid metric: %s\n", metrics_other[j]);
				continue;
			}
			else
			{
				int numPass = cuptiOb.getNumPass();
				if (numPass > 0)
				{
					//printf("\tmetric[%d]: %s , numPass: %d\n", j, metrics_other[j], numPass);
					for (int pass = 0; pass < numPass; pass++) {
						cuptiOb.startNewPass(pass);
						dim3 offset = multiply(epochGridSize, pd++, splitType);
						kernels[i].launch(epochGridSize, offset, splitType);
						if (!kernels[i].isLeft)
							break;
					}
					int stat = cuptiOb.checkAfterPass();
					values_other[j] = cuptiOb.metricGenerateValue();
					printf("\t\tmetric[%d] %s: %llu\n", j, metrics_other[j], (unsigned long long)values_other[j].metricValueUint64);
				}
				else
				{
					printf("\t\tmetric[%d] %s: !!! numPass=0\n", j, metrics_other[j]);
				}
				fflush(stdout);
				
				cuptiOb.unsetup();
			}
			///////////////// test it
		}
		}
	}
	
	// sort
	for (int i = 0;i < nKernel;i++)
	{
		for (int j = 0;j < nKernel - i - 1;j++)
			if ( kernels[j].CtoM > kernels[j + 1].CtoM)
			{
				kernelObj temp = kernels[j];
				kernels[j] = kernels[j + 1];
				kernels[j + 1] = temp;
			}
	}
	if (!_silentVerbos)
		printf("================\n");
	for (int i = 0;i < nKernel;i++)
		if (!_silentVerbos)
			printf("kernels[%d].CtoM=%f, kernels[i].sampleKernelTime=%ld\n", i, kernels[i].CtoM, kernels[i].sampleKernelTime);
	
	cudaProfilerStart();
}
/*
void cudaWrapper::classifyKernelsTemp(int argc, char **argv)
{
	const int metricInstCount = 9;
	const char* metrics_inst[metricInstCount] = { 
		//"flop_count_hp",
		"flop_count_sp",
		"flop_count_dp",
		"flop_count_sp_special",
		"inst_integer",
		"inst_bit_convert",
		"inst_control",
		"inst_compute_ld_st",
		"inst_misc",
		"inst_inter_thread_communication",
	};
	const int metricMemCount = 2;
	const char* metrics_mem[metricMemCount] = { 
		//"shared_store_transactions", // not used
		//"shared_load_transactions", // not used
		"gld_transactions", 
		"gst_transactions", 
		//"dram_read_transactions",
		//"dram_write_transactions",
		//"sysmem_read_transactions",
		//"sysmem_write_transactions",
		};
		
	printf("at least %d blocks (each at least %d threads) of each kernel should be launched for profiling...\n", _SMXs*(metricInstCount+metricMemCount), _CoresPerSMX);
	
	cuptiObj cuptiOb;
	cuptiOb.init();
	cuptiOb.registerCallbacks();
	
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
	
		blockSplitType splitType;
		dim3 epochGridSize;
		dim3 offset;
		
		//calcSplit(28, kernels[i].gridSize, epochGridSize, offset, splitType);
		//epochGridSize = dim3(1,1,1);
		//myKernel2<<<my1Blocks, my1Threads>>>(myKernel1Data, NC1);
		//printf("myKernel2\n");
		calculate_temp<<<hs_dimGrid, hs_dimBlock>>>(paramsHot, 0, 0, 0);//, step, Rz, Ry, Rx);
		cudaDeviceSynchronize();
		cuptiOb.flushAll();
		uint64_t kernelDuration = cuptiOb.kernelDuration;
		//printf("\tfirst run (%d,%d,%d): kernelDuration:%d \n", epochGridSize.x, epochGridSize.y, epochGridSize.z, kernelDuration);
		
		//kernelDuration
		 
		// use cudaEvents instead !!!!!!!!!!!!
		
		int pd = 1;
		unsigned long long n_insts = 0;
		unsigned long long n_mems = 0;

		CUpti_MetricValue values_inst[metricInstCount];
		printf("\treading %d inst metrics\n", metricInstCount);
		for (int j = 0;j < metricInstCount;j++)
		{
			//printf("\nto setup ...");
			if (cuptiOb.setup(metrics_inst[j]) != 0)
			{
				printf("\n\t\tinvalid metric: %s\n", metrics_inst[j]);
				continue;
			}
			else
			{
				int numPass = cuptiOb.getNumPass();
				//printf("\tmetric[%d]: %s , numPass: %d\n", j, metrics_inst[j], numPass);
				for (int pass = 0; pass < numPass; pass++) {
					cuptiOb.startNewPass(pass);
					dim3 offset = multiply(epochGridSize, pd++, splitType);
					//printf("offset(%d,%d)\n", offset.x, offset.y);
					//myKernel2<<<my1Blocks, my1Threads>>>(myKernel1Data, NC1);
					//printf("myKernel2\n");
					calculate_temp<<<hs_dimGrid, hs_dimBlock>>>(paramsHot, 0, 0, 0);//, step, Rz, Ry, Rx);
				}
				int stat = cuptiOb.checkAfterPass();
				values_inst[j] = cuptiOb.metricGenerateValue();
				printf("\t\tmetric[%d] %s: %llu\n", j, metrics_inst[j], (unsigned long long)values_inst[j].metricValueUint64);
				
				n_insts += (unsigned long long)values_inst[j].metricValueUint64;
				cuptiOb.unsetup();
			}
			///////////////// test it
		}

		CUpti_MetricValue values_mem[metricMemCount];
		printf("\treading %d memory metrics\n", metricMemCount);
		for (int j = 0;j < metricMemCount;j++)
		{
			//printf("\nto setup ...");
			if (cuptiOb.setup(metrics_mem[j]) != 0)
			{
				printf("\n\t\tinvalid metric: %s\n", metrics_mem[j]);
				continue;
			}
			else
			{
				int numPass = cuptiOb.getNumPass();
				//printf("\tmetric[%d]: %s , numPass: %d\n", j, metrics_mem[j], numPass);
				for (int pass = 0; pass < numPass; pass++) {
					cuptiOb.startNewPass(pass);
					dim3 offset = multiply(epochGridSize, pd++, splitType);
					//printf("offset(%d,%d)\n", offset.x, offset.y);
					//myKernel2<<<my1Blocks, my1Threads>>>(myKernel1Data, NC1);
					//printf("myKernel2\n");
					calculate_temp<<<hs_dimGrid, hs_dimBlock>>>(paramsHot, 0, 0, 0);//, step, Rz, Ry, Rx);
				}
				int stat = cuptiOb.checkAfterPass();
				values_mem[j] = cuptiOb.metricGenerateValue();
				printf("\t\tmetric[%d] %s: %llu\n", j, metrics_mem[j], (unsigned long long)values_mem[j].metricValueUint64);
				
				n_mems += (unsigned long long)values_mem[j].metricValueUint64;
				cuptiOb.unsetup();
			}
			///////////////// test it
		}
		//double gflops = ((double)n_insts/1000000)/(kernelDuration/1000);
		double ratio = (double)n_insts/(double)n_mems;
		printf("\tn_insts:%llu, n_mems:%llu, ratio:%g\n", n_insts, n_mems, ratio);

	// sort
	
}
*/

void cudaWrapper::launch()
{
	/*********************************************************    Preprocess: profile ans classify   ******************************************************************/
	/*********************************************************    Preprocess: profile ans classify   ******************************************************************/
	/*********************************************************    Preprocess: profile ans classify   ******************************************************************/

//return;
	//printf("launch() here: %d\n", __LINE__);
	int nKernel = min(nKernels, nAddedKernels);
#ifdef __linux__
	cudaStream_t streams[nKernel];
#elif _WIN32
	cudaStream_t *streams;
	streams = new cudaStream_t[nKernel];
#endif
	for (int i = 0;i < nKernel;i++)
		cudaStreamCreate(&streams[i]);
	for (int i = 0;i < nKernel;i++)
	{
		//printf("launching kernel %d\n", i);
		fflush(stdout);
		
		//dim3 ch(kernels[i].gridSize.x/2, kernels[i].gridSize.y, kernels[i].gridSize.z);
		//kernels[i].launch(ch, streams[i], xSplit);
		//dim3 padd(kernels[i].gridSize.x/2, 0, 0);
		//kernels[i].launch(ch, padd, streams[i], xSplit);
		
		
		kernels[i].launch(streams[i]);
	}
	//printf("launch() here: %d\n", __LINE__);
}

void cudaWrapper::launchRandom()
{
	//printf("launch() here: %d\n", __LINE__);
	int nKernel = min(nKernels, nAddedKernels);
	if (!_silentVerbos)
		printf("%d -> nKernels:%d, nAddedKernels:%d, nKernel: %d\n", __LINE__, nKernels, nAddedKernels, nKernel);
#ifdef __linux__
	cudaStream_t streams[nKernel];
#elif _WIN32
	cudaStream_t *streams;
	streams = new cudaStream_t[nKernel];
#endif
	for (int i = 0;i < 2;i++)
	{
		cudaStreamCreate(&streams[i]);
		//streams[i] = 0;
	}
	
	//kernelObj* kernelsMem;
	//kernelObj* kernelsComp;
	
	//int memK = nKernel / 2;
	//int compK = nKernel / 2;
	
	int *kernelIndex = (int*)malloc(sizeof(int)*nKernel);
	for (int i = 0;i < nKernel;i++)
		kernelIndex[i] = i;
	shuffle(kernelIndex, nKernel);
	
	int i, j;
	//printf("%d -> memK: %d, compK: %d\n", __LINE__, memK, compK);
	for (i = 0, j = 1;i < nKernel || j < nKernel;i+=2,j+=2)
	{
		if (j < nKernel)
			if (!_silentVerbos)
				printf ("launching kernel %d in stream 0   ,   launching kernel %d in stream 1\n", kernelIndex[i], kernelIndex[j]);
		else
			if (!_silentVerbos)
				printf ("launching kernel %d in stream 0\n", kernelIndex[i]);
		if (i < nKernel)
			kernels[kernelIndex[i]].launch(streams[0]);
		if (j < nKernel)
			kernels[kernelIndex[j]].launch(streams[1]);
	}
			
	//printf("iteration:%d, chunkLaunchedMem:%d, chunkLaunchedComp:%d\n============ \n", it, chunkLaunchedMem, chunkLaunchedComp);
	//fflush(stdout);
}

void cudaWrapper::launchSerial()
{
	if (!_silentVerbos)
		printf("launchSerial\n");
	int nKernel = min(nKernels, nAddedKernels);
	for (int i = 0;i < nKernel;i++)
	{
		if (!_silentVerbos)
			printf("launching kernel serially %d\n", i);
		fflush(stdout);
		kernels[i].launch();
	}
}

void cudaWrapper::launchConcurrentMemoryCompute()
{
	//printf("launch() here: %d\n", __LINE__);
	int nKernel = min(nKernels, nAddedKernels);
	if (!_silentVerbos)
		printf("%d -> nKernels:%d, nAddedKernels:%d, nKernel: %d\n", __LINE__, nKernels, nAddedKernels, nKernel);
#ifdef __linux__
	cudaStream_t streams[nKernel];
#elif _WIN32
	cudaStream_t *streams;
	streams = new cudaStream_t[nKernel];
#endif
	for (int i = 0;i < nKernel;i++)
	{
		cudaStreamCreate(&streams[i]);
		//streams[i] = 0;
	}
	
	kernelObj* kernelsMem;
	kernelObj* kernelsComp;
	
	int memK = 0;
	int compK = 0;
	for (int i = 0;i < nKernel;i++)
	{
		if (kernels[i].kClass == computeBound)
			compK++;
		else if (kernels[i].kClass == memoryBound)
			memK++;
	}
	//printf("%d -> memK: %d, compK: %d\n", __LINE__, memK, compK);
	
	kernelsMem = (kernelObj*)malloc(memK*sizeof(kernelObj));
	kernelsComp = (kernelObj*)malloc(compK*sizeof(kernelObj));
	
	for (int i = 0, memK = 0, compK = 0;i < nKernel;i++)
	{
		//printf("%d -> kernels[i].kClass:%d\n", __LINE__, kernels[i].kClass);
		if (kernels[i].kClass == computeBound)
			kernelsComp[compK++] = kernels[i];
		else if (kernels[i].kClass == memoryBound)
			kernelsMem[memK++] = kernels[i];
	}

	
	int ratioMem = 4;
	int ratioComp = 4;
	int i, j;
	//printf("%d -> memK: %d, compK: %d\n", __LINE__, memK, compK);
	for (i = 0, j = 0;i < memK && j < compK;i++,j++)
	{
		// check whether kernels have more blocks than 32, 8?		
		
		
		int runMeM = 32;
		int runComp = 8;
		
		blockSplitType splitMem = xSplit;
		blockSplitType splitComp = xSplit;

		if (!_silentVerbos)
			printf("%d -> mem: grid(%d,%d)\n", __LINE__, kernelsMem[i].gridSize.x, kernelsMem[i].gridSize.y);
		if (!_silentVerbos)
			printf("%d -> comp: grid(%d,%d)\n", __LINE__, kernelsComp[j].gridSize.x, kernelsComp[j].gridSize.y);
		
		// whether to split blocks in x coordinate or y for each of two kernels
		/**************************************** Memory-Bound ****************************************************/
		dim3 epochGridSizeMem(runMeM, kernelsMem[i].gridSize.y, kernelsMem[i].gridSize.z);
		dim3 offsetMem(runMeM, 0, 0);
		calcSplit(runMeM, kernelsMem[i].gridSize, epochGridSizeMem, offsetMem, splitMem);		
		/**************************************** Memory-Bound ****************************************************/
		
		// if there is something wrong, maybe it's related to the new function `calcSplit`
		
		/**************************************** Compute-Bound ****************************************************/
		dim3 epochGridSizeComp(runComp, kernelsComp[j].gridSize.y, kernelsComp[j].gridSize.z);
		dim3 offsetComp(runComp, 0, 0);
		calcSplit(runComp, kernelsComp[j].gridSize, epochGridSizeComp, offsetComp, splitComp);
		/**************************************** Compute-Bound ****************************************************/
		int chunkLaunchedMem = 0;
		int chunkLaunchedComp = 0;
		for (int it = 0;kernelsComp[j].isLeft || kernelsMem[i].isLeft;it++)
		{
			if (!_silentVerbos)
				printf("%d -> iteration: %d, kernelsComp[j].isLeft: %d, kernelsMem[i].isLeft: %d\n", __LINE__, it, kernelsComp[j].isLeft, kernelsMem[i].isLeft);
			// memory-bound kernels are launched in streams[0] and 
			// compute-bound kernels are launched in streams[1]

			if (kernelsMem[i].isLeft)
			{
				for (int itM = 0;itM < ratioMem;itM++)
				{
					if (kernelsMem[i].isLeft)
					{
						dim3 offset = multiply(offsetMem, chunkLaunchedMem, splitMem);
						kernelsMem[i].launch(epochGridSizeMem, offset, streams[0], splitMem);
						chunkLaunchedMem++;
					}
				}
			}
			
			if (kernelsComp[j].isLeft)
			{
				for (int itC = 0;itC < ratioComp;itC++)
				{
					if (kernelsComp[j].isLeft)
					{
						dim3 offset = multiply(offsetComp, chunkLaunchedComp, splitComp);
						kernelsComp[j].launch(epochGridSizeComp, offset, streams[1], splitComp);
						chunkLaunchedComp++;
					}
				}
			}
			
			if (!_silentVerbos)
				printf("iteration:%d, chunkLaunchedMem:%d, chunkLaunchedComp:%d\n============ \n", it, chunkLaunchedMem, chunkLaunchedComp);
			fflush(stdout);
		}
	}
}

void cudaWrapper::launchConcurrentMemoryComputeNoSlice()
{
	//printf("launch() here: %d\n", __LINE__);
	int nKernel = min(nKernels, nAddedKernels);
	if (!_silentVerbos)
		printf("%d -> nKernels:%d, nAddedKernels:%d, nKernel: %d\n", __LINE__, nKernels, nAddedKernels, nKernel);
#ifdef __linux__
	cudaStream_t streams[nKernel];
#elif _WIN32
	cudaStream_t *streams;
	streams = new cudaStream_t[nKernel];
#endif
	for (int i = 0;i < nKernel;i++)
	{
		cudaStreamCreate(&streams[i]);
		//streams[i] = 0;
	}
	
	kernelObj* kernelsMem;
	kernelObj* kernelsComp;
	
	int memK = 0;
	int compK = 0;
	for (int i = 0;i < nKernel;i++)
	{
		if (kernels[i].kClass == computeBound)
			compK++;
		else if (kernels[i].kClass == memoryBound)
			memK++;
	}
	//printf("%d -> memK: %d, compK: %d\n", __LINE__, memK, compK);
	
	kernelsMem = (kernelObj*)malloc(memK*sizeof(kernelObj));
	kernelsComp = (kernelObj*)malloc(compK*sizeof(kernelObj));
	
	for (int i = 0, memK = 0, compK = 0;i < nKernel;i++)
	{
		//printf("%d -> kernels[i].kClass:%d\n", __LINE__, kernels[i].kClass);
		if (kernels[i].kClass == computeBound)
			kernelsComp[compK++] = kernels[i];
		else if (kernels[i].kClass == memoryBound)
			kernelsMem[memK++] = kernels[i];
	}

	
	//int ratioMem = 4;
	//int ratioComp = 4;
	int i, j;
	//printf("%d -> memK: %d, compK: %d\n", __LINE__, memK, compK);
	for (i = 0, j = 0;i < memK || j < compK;i++,j++)
	{
		if (i < memK)
		{
			if (!_silentVerbos)
				printf ("launching kernel mem[%d] in stream 0\n", i);
			kernelsMem[i].launch(streams[0]);
			if (!_silentVerbos)
				printf("%d -> mem: grid(%d,%d)\n", __LINE__, kernelsMem[i].gridSize.x, kernelsMem[i].gridSize.y);
		}
		if (j < compK)
		{
			kernelsComp[j].launch(streams[1]);
			if (!_silentVerbos)
				printf("%d -> comp: grid(%d,%d)\n", __LINE__, kernelsComp[j].gridSize.x, kernelsComp[j].gridSize.y);
			if (!_silentVerbos)
				printf ("launching kernel comp[%d] in stream 1\n", j);
		}
		
	}
}

void cudaWrapper::launchConcurrent()
{
	int nKernel = min(nKernels, nAddedKernels);

	if (!_silentVerbos)
		printf("\n\n\n================================\n");
	if (!_silentVerbos)
		printf("Kernel set info:\nnKernel: %d\n", nKernel);
	for (int i = 0;i < nKernel;i++)
	{
		if (!_silentVerbos)
			printf("Kernel %i: id: %i, Grid(%d,%d,%d), Block(%d,%d,%d)\n", i, kernels[i].id, kernels[i].gridSize.x, kernels[i].gridSize.y, kernels[i].gridSize.z,
				kernels[i].blockSize.x, kernels[i].blockSize.y, kernels[i].blockSize.z);
	}
	if (!_silentVerbos)
		printf("================================\n\n\n");
	/*********************************************************    Preprocess: profile ans classify   ******************************************************************/
	/*********************************************************    Preprocess: profile ans classify   ******************************************************************/
	/*********************************************************    Preprocess: profile ans classify   ******************************************************************/
	
	
	
	//cudaProfilerStart();

	/*
	cudaStream_t streams[nKernel];
	for (int i = 0;i < nKernel;i++)
		cudaStreamCreate(&streams[i]);
	*/

	cudaStream_t streams[2];
	for (int i = 0;i < 2;i++)
		cudaStreamCreate(&streams[i]);
	
	dim3 epochGridSizeC, epochGridSizeM, offsetC, offsetM;
	int blocksToRunC, blocksToRunM;
	int ic, im;
	blockSplitType splitTypeC, splitTypeM;

	//classifyKernels();
	executionTimeCalc();

	/*
	for (int i = 0;i < nKernel;i++)
	{
		dim3 epochGridSize, offset;
		blockSplitType splitType;
		blocksToRunC = blocksToRunCalc(kernels[i].blockSize);
		calcSplit(blocksToRunC, kernels[i].gridSize, epochGridSize, offset, splitType);

		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);
		kernels[i].launch(epochGridSize, splitType);
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		kernels[i].sampleKernelTimeFloat = msecTotal;
		printf("msec: %f\n", msecTotal);
	}
	*/

	if (!_silentVerbos)
		printf("\n\n\n================================\n");
	if (!_silentVerbos)
		printf("Start of iterative launch\n\n");
	for (ic = 0, im = nKernel - 1;ic < im;ic++,im--)
	{
		if (!_silentVerbos)
			printf("kernels[ic].id: %d, kernels[im].id:%d\n", kernels[ic].id, kernels[im].id);
		int pdC = 1, pdM = 1;
		blocksToRunC = blocksToRunCalc(kernels[ic].blockSize);
		calcSplit(blocksToRunC, kernels[ic].gridSize, epochGridSizeC, offsetC, splitTypeC);
		
		blocksToRunM = blocksToRunCalc(kernels[im].blockSize);
		calcSplit(blocksToRunM, kernels[im].gridSize, epochGridSizeM, offsetM, splitTypeM);
		
		int ratio = 1;
		bool memIsLonger = kernels[ic].sampleKernelTimeFloat < kernels[im].sampleKernelTimeFloat;
		ratio = memIsLonger ? kernels[im].sampleKernelTimeFloat / kernels[ic].sampleKernelTimeFloat : kernels[ic].sampleKernelTimeFloat / kernels[im].sampleKernelTimeFloat;
		if (!_silentVerbos)
			printf(COLOR_GREEN);
		if (!_silentVerbos)
			printf("[im] is longer:%d, ratio: %d\n", memIsLonger?1:0, ratio);
		if (!_silentVerbos)
			printf(COLOR_RESET);
		
		int phase = 0;
		int phase_ic = 0;
		int phase_im = 0;
		bool nextImAtOnce = false;
		bool nextIcAtOnce = false;
		while (kernels[ic].isLeft || kernels[im].isLeft)
		{
			//printf("--------------\n");
			if (kernels[ic].isLeft)
			{
				offsetC = multiply(epochGridSizeC, pdC++, splitTypeC);
				bool changeNextIcAtOnce = false;
				if (!kernels[im].isLeft && ic == im - 1)
					changeNextIcAtOnce = true;
				if (!nextIcAtOnce)
				{
					kernels[ic].launch(epochGridSizeC, offsetC, streams[0], splitTypeC);
					phase_ic++;
					if (memIsLonger && ratio > 1)
					{
						for (int rep = 0; rep < ratio - 1; rep++)
						{
							if (!kernels[ic].isLeft)
								break;
							offsetC = multiply(epochGridSizeC, pdC++, splitTypeC);
							kernels[ic].launch(epochGridSizeC, offsetC, streams[0], splitTypeC);
							phase_ic++;
						}
					}
				}
				else
				{
					kernels[ic].launchLeft(offsetC, streams[0]);
					phase_ic++;
				}
				if (changeNextIcAtOnce)
				{
					nextIcAtOnce = true;
					changeNextIcAtOnce = false;
				}

			}
			if (kernels[im].isLeft)
			{
				offsetM = multiply(epochGridSizeM, pdM++, splitTypeM);
				bool changeNextImAtOnce = false;
				if (!kernels[ic].isLeft && ic == im - 1)
					changeNextImAtOnce = true;
				if (!nextImAtOnce)
				{
					kernels[im].launch(epochGridSizeM, offsetM, streams[1], splitTypeM);
					phase_im++;
					if (!memIsLonger && ratio > 1)
					{
						for (int rep = 0; rep < ratio - 1; rep++)
						{
							if (!kernels[im].isLeft)
								break;
							offsetM = multiply(epochGridSizeM, pdM++, splitTypeM);
							kernels[im].launch(epochGridSizeM, offsetM, streams[1], splitTypeM);
							phase_im++;
						}
					}
				}
				else
				{
					kernels[im].launchLeft(offsetM, streams[1]);
					phase_im++;
				}
				if (changeNextImAtOnce)
				{
					nextImAtOnce = true;
					changeNextImAtOnce = false;
				}
			}
			phase++;
			
			////////////////////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			/*
			if (phase >= 10) {
				printf(COLOR_RED);
				printf("%d: temp return on phase: %d, phase_ic: %d, phase_im: %d\n", __LINE__, phase, phase_ic, phase_im); 
				printf(COLOR_RESET);
				return;
			}
			*/
			
		}
		if (!_silentVerbos)
			printf("%d: while loop finished after phase: %d, phase_ic: %d, phase_im: %d\n\n\n", __LINE__, phase, phase_ic, phase_im); 
		
		//kernels[i].launch(epochGridSize, offset, splitType);
		//break;
	}
	if (ic == im)
	{
		if (kernels[ic].isLeft)
		{
			cudaStream_t streamNew;
			cudaStreamCreate(&streamNew);
			kernels[ic].launch(streamNew);
		}
		if (!_silentVerbos)
			printf("%d: last kernel run single: %d\n\n\n", __LINE__, ic); 
	}
	if (!_silentVerbos)
		printf("---------------------\n");
	for (int i =0;i < nKernel;i++)
		kernels[i].echoRunningInfo();
	if (!_silentVerbos)
		printf("---------------------\n");
	if (!_silentVerbos)
		printf("End of iterative launch\n================================\n\n\n");
	
}

void cudaWrapper::launchConcurrentSimple()
{
	int nKernel = min(nKernels, nAddedKernels);
	for (int i = 0;i < nKernel;i++)
	{
		cudaStreamCreate(&streams[i]);
		kernels[i].launch(streams[i]);
	}
	
}