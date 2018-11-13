//#include <ksh_runtime.cu>

__global__ void k1(kernelParams params)
{
	dim3 blockIdx(rebuildBlock(params.offset));
	int n = params.getParameter<int>(0);
	//printf("k1(): n: %d\n", n);
	float* d_f = (float*)(params.getParameter(1));
	d_f[threadIdx.x]++;
	//printf("k1(): d_f[%d]: %f\n", threadIdx.x, d_f[threadIdx.x]);
}

__global__ void k2(kernelParams params)
{
	dim3 blockIdx(rebuildBlock(params.offset));
	int n = params.getParameter<int>(0);
	//printf("k2(): n: %d\n", n);
	float* d_f = (float*)(params.getParameter(1));
	//printf("k2(): d_f[%d]: %f\n", threadIdx.x, d_f[threadIdx.x]);
}

kernelParams prepare1(cudaWrapper &wrapper)
{
	kernelParams params1(3);	
	int n1 = 6;
    float* fff1 = (float *)malloc(n1*sizeof(float));
    for (int i = 0;i < n1;i++)
        fff1[i] = i*100 + i/(float)100;
    float* d_fff1;
    (float*)cudaMalloc((void **)&d_fff1, n1*sizeof(float));
    cudaMemcpy(d_fff1, fff1, n1*sizeof(float), cudaMemcpyHostToDevice);
	params1.addParameter<int>(n1);
	params1.addParameter((void*)d_fff1);
	params1.setPadding(dim3(300,400));
	wrapper.addKernel((kernelPtr)k1, params1, dim3(1,1), dim3(n1,1));
	return params1;
}	

kernelParams prepare2(cudaWrapper &wrapper)
{
	kernelParams params2(2);	
	int n2 = 4;
    float* fff2 = (float *)malloc(n2*sizeof(float));
    for (int i = 0;i < n2;i++)
        fff2[i] = i*10 + i/(float)10;
    float* d_fff2;
    (float*)cudaMalloc((void **)&d_fff2, n2*sizeof(float));
    cudaMemcpy(d_fff2, fff2, n2*sizeof(float), cudaMemcpyHostToDevice);
	params2.addParameter<int>(n2);
	params2.addParameter((void*)d_fff2);
	params2.setPadding(dim3(30,40));
	wrapper.addKernel((kernelPtr)k2, params2, dim3(1,1), dim3(n2,1));
	return params2;
}	
