#include <cuda_runtime.h>
#include <string.h>
#include <string>
#include <stdio.h>
#include <iostream>
using namespace std;

#define SEPARATOR ","

struct eventInfo {
	int kernelID;
	cudaEvent_t start;
	cudaEvent_t stop;
	float msec;
	float msecStartFromStart;
	float msecStopFromStart;
	dim3 blocks;
	dim3 threads;
};

void setEventParams(eventInfo& info, dim3 blocks, dim3 threads)
{
	info.blocks = blocks;
	info.threads = threads;
}

void setEventParams(eventInfo& info, int kernelID, dim3 blocks, dim3 threads)
{
	info.kernelID = kernelID;
	info.blocks = blocks;
	info.threads = threads;
}
eventInfo createEventInfo(int kernelID, dim3 blocks, dim3 threads)
{
	eventInfo info;
	info.kernelID = kernelID;
	cudaEventCreate(&info.start);
	cudaEventCreate(&info.stop);
	info.msec = 0;
	info.msecStartFromStart = 0;
	info.msecStopFromStart = 0;
	info.blocks = blocks;
	info.threads = threads;
	return info;
}

struct streamEventInfo {
	int streamID;
	int partitionCount;
	eventInfo* events;
};

streamEventInfo createStreamEventInfo(int kernelID, int streamID, int partitionCount, dim3 blocks, dim3 threads)
{
	//printf("create\n");
	streamEventInfo info;
	info.streamID = streamID;
	info.partitionCount = partitionCount;
	int size = partitionCount * sizeof(eventInfo);
	info.events = (eventInfo*)malloc(size);
	for (int i = 0;i < partitionCount;i++)
		info.events[i] = createEventInfo(kernelID, blocks, threads);
	
	return info;
}

streamEventInfo createStreamEventInfo(int kernelID, int streamID, int partitionCount, int blocksN, int threadsN)
{
	//printf("create\n");
	streamEventInfo info;
	info.streamID = streamID;
	info.partitionCount = partitionCount;
	int size = partitionCount * sizeof(eventInfo);
	info.events = (eventInfo*)malloc(size);
	dim3 blocks(blocksN, blocksN == 0 ? 0 : 1, blocksN == 0 ? 0 : 1);
	dim3 threads(threadsN, threadsN == 0 ? 0 : 1, threadsN == 0 ? 0 : 1);
	for (int i = 0;i < partitionCount;i++)
		info.events[i] = createEventInfo(kernelID, blocks, threads);
	
	return info;
}

void writeEventRecord(ofstream& fLog, eventInfo info, bool isNextParam, int nextParam)
{
	fLog << info.kernelID << SEPARATOR << info.blocks.x << SEPARATOR << info.blocks.y << SEPARATOR << info.blocks.z << SEPARATOR;
	fLog << info.threads.x << SEPARATOR << info.threads.y << SEPARATOR << info.threads.z << SEPARATOR;
	fLog << info.start << SEPARATOR << info.stop << SEPARATOR << info.msec;
	if (isNextParam)
		fLog << SEPARATOR << nextParam;
	fLog << "\n";
}

void writeStreamEventRecord(ofstream& fLog, streamEventInfo info, int eventIX, bool isNextParam, int nextParam)
{
	fLog << info.events[eventIX].kernelID << SEPARATOR;
	fLog << info.events[eventIX].blocks.x << SEPARATOR << info.events[eventIX].blocks.y << SEPARATOR << info.events[eventIX].blocks.z << SEPARATOR;
	fLog << info.events[eventIX].threads.x << SEPARATOR << info.events[eventIX].threads.y << SEPARATOR << info.events[eventIX].threads.z << SEPARATOR;
	fLog << info.streamID << SEPARATOR << info.events[eventIX].start << SEPARATOR << info.events[eventIX].stop << SEPARATOR << info.events[eventIX].msec << SEPARATOR;
	fLog << info.events[eventIX].msecStartFromStart << SEPARATOR << info.events[eventIX].msecStopFromStart << SEPARATOR << (info.events[eventIX].msecStopFromStart-info.events[eventIX].msecStartFromStart);
	if (isNextParam)
		fLog << SEPARATOR << nextParam;
	fLog << "\n";
}

streamEventInfo createStreamEventInfo(int streamID, int partitionCount)
{
	streamEventInfo info;
	info.streamID = streamID;
	info.partitionCount = partitionCount;
	int size = partitionCount * sizeof(eventInfo);
	info.events = (eventInfo*)malloc(size);
	for (int i = 0;i < partitionCount;i++)
	{
		cudaEventCreate(&info.events[i].start);
		cudaEventCreate(&info.events[i].stop);
	}
	
	return info;
}

struct streamSummaryInfo {
	int partitionCount;
	cudaEvent_t start;
	cudaEvent_t stop;
	int streamID;
	float sumExecutionTime;
	float msec;
};
	
