enum kernelClass {
	computeBound = 1,
	memoryBound = 2,
};

enum blockSplitType {
	noSplit = 0,
	xSplit = 1,
	ySplit = 2,
	zSplit = 4,
};

__device__   __forceinline__  dim3 rebuildBlock(dim3 blockOffset)
{
	int newX = blockOffset.x + blockIdx.x;
	int newY = blockOffset.y + blockIdx.y;
	int newZ = blockOffset.z + blockIdx.z;
	dim3 blockIdx(newX, newY, newZ);
	return blockIdx;
}

__device__   __forceinline__  dim3 rebuildBlock(int x, int y, int z)
{
	int newX = x + blockIdx.x;
	int newY = y + blockIdx.y;
	int newZ = z + blockIdx.z;
	dim3 blockIdx(newX, newY, newZ);
	return blockIdx;
}

dim3 multiply(dim3 d, int mult)
{
	dim3 newd(d.x * mult, d.y * mult, d.z * mult);
	//printf("d(%d,%d,%d), newd(%d,%d,%d)\n", d.x, d.y, d.z, newd.x, newd.y, newd.z);
	return dim3(d.x * mult, d.y * mult, d.z * mult);
}

dim3 multiply(dim3 d, int mult, blockSplitType type)
{
	dim3 newd	(type == xSplit ? d.x * mult : d.x, 
				 type == ySplit ? d.y * mult : d.y, 
				 type == zSplit ? d.z * mult : d.z);
	//printf("mult:%d, splitType:%d, d(%d,%d,%d), newd(%d,%d,%d)\n", mult, type, d.x, d.y, d.z, newd.x, newd.y, newd.z);
	return newd;
}

dim3 min(dim3 d1, dim3 d2)
{
	return dim3(min(d1.x, d2.x), min(d1.y, d2.y), min(d1.z, d2.z));
}

int Max(int a, int b)
{
	return a > b ? a : b;
}