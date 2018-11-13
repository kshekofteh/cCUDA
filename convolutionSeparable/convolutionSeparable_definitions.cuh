
#include "convolutionSeparable_common.h"


#define INIT_convolutionSpearable()\
\
    float *h_Kernel,*h_Input,*h_Buffer,*h_OutputCPU,*h_OutputGPU; \
\
    float *d_Input,*d_Output,*d_Buffer;\
\
    int imageW = 3072;\
    int imageH = 3072;\
    int iterations = 16;\
    if (checkCmdLineFlag(argc, (const char **)argv, "imageW"))\
    {\
        imageW = getCmdLineArgumentInt(argc, (const char **)argv, "imageW");\
    }\
    if (checkCmdLineFlag(argc, (const char **)argv, "imageH"))\
    {\
        imageH = getCmdLineArgumentInt(argc, (const char **)argv, "imageH");\
    }\
    if (checkCmdLineFlag(argc, (const char **)argv, "iterations"))\
    {\
        iterations = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");\
    }\
\
    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);\
    printf("Allocating and initializing host arrays...\n");\
    h_Kernel    = (float *)malloc(KERNEL_LENGTH * sizeof(float));\
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));\
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));\
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));\
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));\
    srand(200);\
\
    for (unsigned int i = 0; i < KERNEL_LENGTH; i++)\
    {\
        h_Kernel[i] = (float)(rand() % 16);\
    }\
\
    for (unsigned i = 0; i < imageW * imageH; i++)\
    {\
        h_Input[i] = (float)(rand() % 16);\
    }\
\
    printf("Allocating and initializing CUDA arrays...\n");\
    checkCudaErrors(cudaMalloc((void **)&d_Input,   imageW * imageH * sizeof(float)));\
    checkCudaErrors(cudaMalloc((void **)&d_Output,  imageW * imageH * sizeof(float)));\
    checkCudaErrors(cudaMalloc((void **)&d_Buffer , imageW * imageH * sizeof(float)));\
\
    setConvolutionKernel(h_Kernel);\
    checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));\
\
    printf("Running GPU convolution (%u identical iterations)...\n\n", iterations);\
\
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);\
    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);\
    assert(imageH % ROWS_BLOCKDIM_Y == 0);\
\
    dim3 convBlocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);\
    dim3 convThreads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);\
	