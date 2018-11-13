/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
* This sample implements 64-bin histogram calculation
* of arbitrary-sized 8-bit data array
*/

// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// project include
#include "histogram_common.h"

const int numRuns = 1;
const static char *sSDKsample = "[histogram]\0";

int main(int argc, char **argv)
{
    uchar *h_histogramData;
    uint  *h_HistogramCPU, *h_HistogramGPU;
    uchar *d_histogramData;
    uint  *d_Histogram;
    int PassFailFlag = 1;
    unsigned long int byteCount = 64 * 1048576; // = 67108864
    if (checkCmdLineFlag(argc, (const char **)argv, "bC"))
    {
        byteCount = getCmdLineArgumentInt(argc, (const char **)argv, "bC");
    }
    uint uiSizeMult = 1;
    // Optional Command-line multiplier to increase size of array to histogram
    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
    {
        uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
        uiSizeMult = MAX(1,MIN(uiSizeMult, 10));
        byteCount *= uiSizeMult;
    }

    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    h_histogramData         = (uchar *)malloc(byteCount);
    h_HistogramCPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
    h_HistogramGPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));

    printf("...generating histogram input data\n");
    srand(2009);

    for (uint i = 0; i < byteCount; i++)
    {
        h_histogramData[i] = rand() % 256;
    }

    printf("...allocating GPU memory and copying input data\n\n");
    checkCudaErrors(cudaMalloc((void **)&d_histogramData, byteCount));
    checkCudaErrors(cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_histogramData, h_histogramData, byteCount, cudaMemcpyHostToDevice));

    {
        printf("Initializing 256-bin histogram...\n");
        initHistogram256();

        printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);

        for (int iter = 0; iter < numRuns; iter++)
        {
            //iter == -1 -- warmup iteration
            if (iter == 0)
            {
                checkCudaErrors(cudaDeviceSynchronize());
            }

            histogram256(d_Histogram, d_histogramData, byteCount);
        }

        cudaDeviceSynchronize();

        printf("\nValidating GPU results...\n");
        printf(" ...reading back GPU results\n");
        checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

        printf(" ...histogram256CPU()\n");
        histogram256CPU(
            h_HistogramCPU,
            h_histogramData,
            byteCount
        );

        printf(" ...comparing the results\n");

        for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
            if (h_HistogramGPU[i] != h_HistogramCPU[i])
            {
                PassFailFlag = 0;
            }

        printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");

        printf("Shutting down 256-bin histogram...\n\n\n");
        closeHistogram256();
    }

    printf("Shutting down...\n");
    checkCudaErrors(cudaFree(d_Histogram));
    checkCudaErrors(cudaFree(d_histogramData));
    free(h_HistogramGPU);
    free(h_HistogramCPU);
    free(h_histogramData);

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

    printf("%s - Test Summary\n", sSDKsample);

    // pass or fail (for both 64 bit and 256 bit histograms)
    if (!PassFailFlag)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
