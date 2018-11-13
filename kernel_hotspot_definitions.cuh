
#define BLOCK_SIZE_HOTSPOT 16 

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline

#define INIT_hotspot() \
    int hs_size;\
    int grid_rows=0,grid_cols=0;\
    float *FilesavingTemp,*FilesavingPower,*MatrixOut; \
    char *tfile, *pfile, *ofile;\
\
    int total_iterations = 60;\
    int pyramid_height = 1; \
\
    if (checkCmdLineFlag(argc, (const char **)argv, "grid_rows"))\
        grid_rows = getCmdLineArgumentInt(argc, (const char **)argv, "grid_rows");\
    if (checkCmdLineFlag(argc, (const char **)argv, "grid_cols"))\
        grid_cols = getCmdLineArgumentInt(argc, (const char **)argv, "grid_cols");\
    if (checkCmdLineFlag(argc, (const char **)argv, "pyramid_height"))\
        pyramid_height = getCmdLineArgumentInt(argc, (const char **)argv, "pyramid_height");\
    if (checkCmdLineFlag(argc, (const char **)argv, "total_iterations"))\
        total_iterations = getCmdLineArgumentInt(argc, (const char **)argv, "total_iterations");\
\
    if (checkCmdLineFlag(argc, (const char **)argv, "ofile"))\
		getCmdLineArgumentString(argc, (const char **)argv, "ofile", &ofile);\
\
    if (checkCmdLineFlag(argc, (const char **)argv, "tfile"))\
		getCmdLineArgumentString(argc, (const char **)argv, "tfile", &tfile);\
\
	if (checkCmdLineFlag(argc, (const char **)argv, "pfile"))\
		getCmdLineArgumentString(argc, (const char **)argv, "pfile", &pfile);\
\
    hs_size=grid_rows*grid_cols;\
    int borderCols = (pyramid_height)*EXPAND_RATE/2;\
    int borderRows = (pyramid_height)*EXPAND_RATE/2;\
    int smallBlockCol = BLOCK_SIZE_HOTSPOT-(pyramid_height)*EXPAND_RATE;\
    int smallBlockRow = BLOCK_SIZE_HOTSPOT-(pyramid_height)*EXPAND_RATE;\
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);\
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);\
\
    dim3 hs_dimBlock(BLOCK_SIZE_HOTSPOT, BLOCK_SIZE_HOTSPOT);\
    dim3 hs_dimGrid(blockCols, blockRows);  \
	printf("hs hs_dimBlock(%d,%d)\n", hs_dimBlock.x, hs_dimBlock.y);\
	printf("hs hs_dimGrid(%d,%d)\n", hs_dimGrid.x, hs_dimGrid.y);\
\
	float grid_height = chip_height / grid_rows;\
	float grid_width = chip_width / grid_cols;\
\
	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;\
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);\
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);\
	float Rz = t_chip / (K_SI * grid_height * grid_width);\
\
	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);\
	float step = PRECISION / max_slope;\
    float time_elapsed;\
	time_elapsed=0.001;\
\
    int src = 0, dst = 1;\
\
\
\
\
    FilesavingTemp = (float *) malloc(hs_size*sizeof(float));\
    FilesavingPower = (float *) malloc(hs_size*sizeof(float));\
    MatrixOut = (float *) calloc (hs_size, sizeof(float));\
\
    if( !FilesavingPower || !FilesavingTemp || !MatrixOut)\
        fatal("unable to allocate memory");\
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);\
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);\
\
    float *MatrixTemp[2], *MatrixPower;\
    cudaMalloc((void**)&MatrixTemp[0], sizeof(float)*hs_size);\
    cudaMalloc((void**)&MatrixTemp[1], sizeof(float)*hs_size);\
    cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(float)*hs_size, cudaMemcpyHostToDevice);\
    cudaMalloc((void**)&MatrixPower, sizeof(float)*hs_size);\
    cudaMemcpy(MatrixPower, FilesavingPower, sizeof(float)*hs_size, cudaMemcpyHostToDevice);\

