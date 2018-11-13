#include <helper_string.h>    // helper for string parsing
#include <helper_image.h>     // helper for image and data comparison
#include <helper_cuda.h>      // helper for cuda error checking functions

// Each block transposes/copies a tile of TILE_DIM x TILE_DIM elements
// using TILE_DIM x BLOCK_ROWS threads, so that each thread transposes
// TILE_DIM/BLOCK_ROWS elements.  TILE_DIM must be an integral multiple of BLOCK_ROWS

#define TILE_DIM    16
#define BLOCK_ROWS  16

// This sample assumes that MATRIX_SIZE_X = MATRIX_SIZE_Y
int MATRIX_SIZE_X = 1024;
int MATRIX_SIZE_Y = 1024;
int MUL_FACTOR    = TILE_DIM;

#define FLOOR(a,b) (a-(a%b))

// Compute the tile size necessary to illustrate performance cases for SM20+ hardware
int MAX_TILES = (FLOOR(MATRIX_SIZE_X,512) * FLOOR(MATRIX_SIZE_Y,512)) / (TILE_DIM *TILE_DIM);

// Number of repetitions used for timing.  Two sets of repetitions are performed:
// 1) over kernel launches and 2) inside the kernel over just the loads and stores

#define NUM_REPS  1

// -------------------------------------------------------
// Transposes
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

/*
__global__ void transposeNaive(float *odata, float *idata, int width, int height)
{
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in  = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        odata[index_out+i] = idata[index_in+i*width];
    }
}
*/

__global__ void transposeNaive(kernelParams params, int offsetX, int offsetY, int offsetZ, dim3 gridDim)
{
	float *odata = (float*)(params.getParameter(0));
	float *idata = (float*)(params.getParameter(1));
	int width = params.getParameter<int>(2);
	int height = width;
	
	/****************************************************************/
	// rebuild blockId
	dim3 blockIdx = rebuildBlock(offsetX, offsetY, offsetZ);
	/****************************************************************/
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in  = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        odata[index_out+i] = idata[index_in+i*width];
    }
}


#define INIT_transpose() \
    int trnsp_size;\
    if (checkCmdLineFlag(argc, (const char **)argv, "trnsp_size"))\
        trnsp_size = getCmdLineArgumentInt(argc, (const char **)argv, "trnsp_size");\
	printf("trnsp_size: %d\n", trnsp_size);\
\
    if (trnsp_size % TILE_DIM != 0)\
    {\
        printf("Matrix size must be integral multiple of tile size\nExiting...\n\n");\
        exit(EXIT_FAILURE);\
    }\
    dim3 trnsp_grid(trnsp_size/TILE_DIM, trnsp_size/TILE_DIM), trnsp_threads(TILE_DIM,BLOCK_ROWS);\
\
    if (trnsp_grid.x < 1 || trnsp_grid.y < 1)\
    {\
        printf("trnsp_grid size computation incorrect in test \nExiting...\n\n");\
        exit(EXIT_FAILURE);\
    }\
    size_t trnsp_mem_size = static_cast<size_t>(sizeof(float) * trnsp_size*trnsp_size);\
\
    float *h_idata = (float *) malloc(trnsp_mem_size);\
    float *h_odata = (float *) malloc(trnsp_mem_size);\
\
    float *d_idata, *d_odata;\
    checkCudaErrors(cudaMalloc((void **) &d_idata, trnsp_mem_size));\
    checkCudaErrors(cudaMalloc((void **) &d_odata, trnsp_mem_size));\
\
    for (int i = 0; i < (trnsp_size*trnsp_size); ++i)\
        h_idata[i] = (float) i;\
\
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, trnsp_mem_size, cudaMemcpyHostToDevice));\
    printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n", trnsp_size, trnsp_size, trnsp_grid.x, trnsp_grid.y, TILE_DIM, TILE_DIM, trnsp_threads.x, trnsp_threads.y);
