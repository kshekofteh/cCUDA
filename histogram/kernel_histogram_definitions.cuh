
#define INIT_histogram() \
    uchar *h_histogramData;\
    uint  *h_HistogramCPU, *h_HistogramGPU;\
    uchar *d_histogramData;\
    uint  *d_Histogram;\
    int PassFailFlag = 1;\
    unsigned long int byteCount = 4 * 1048576; \
    if (checkCmdLineFlag(argc, (const char **)argv, "bC"))\
    {\
        byteCount = getCmdLineArgumentInt(argc, (const char **)argv, "bC");\
    }\
    uint uiSizeMult = 1;\
\
    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))\
    {\
        uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");\
		printf("sizemult: %d\n", uiSizeMult);\
        uiSizeMult = MAX(1,MIN(uiSizeMult, 65));\
        byteCount *= uiSizeMult;\
    }\
\
    printf("Initializing data...\n");\
    printf("...allocating CPU memory.\n");\
    h_histogramData         = (uchar *)malloc(byteCount);\
    h_HistogramCPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));\
    h_HistogramGPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));\
\
    printf("...generating histogram input data\n");\
    srand(2009);\
\
    for (uint i = 0; i < byteCount; i++)\
    {\
        h_histogramData[i] = rand() % 256;\
    }\
\
    printf("...allocating GPU memory and copying input data\n\n");\
    checkCudaErrors(cudaMalloc((void **)&d_histogramData, byteCount));\
    checkCudaErrors(cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));\
    checkCudaErrors(cudaMemcpy(d_histogramData, h_histogramData, byteCount, cudaMemcpyHostToDevice));\
\
    printf("Running 256-bin GPU histogram for %u bytes ...\n\n", byteCount);\
	\
	dim3 histogramBlocks(PARTIAL_HISTOGRAM256_COUNT,1,1);\
	dim3 histogramThreads(HISTOGRAM256_THREADBLOCK_SIZE,1,1);\
	\
	assert(byteCount % sizeof(uint) == 0);\
	\
	printf("LAUNCH histogram HERE: %d, blocks: %d, threads per block: %d\n", __LINE__, histogramBlocks.x, histogramThreads.x);\
		
