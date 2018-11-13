#include <cupti.h>
#include <ksh_cupti_api.cuh>
#include <ksh_cupti_callbacks.cuh>

class cuptiObj {
	private:
		CUcontext context;
		CUdevice device;
		CUpti_MetricID metricId;
		CUpti_EventGroupSets *passData;
		MetricData_t metricData;
		unsigned int pass;
		CUpti_SubscriberHandle subscriber;
		CUpti_MetricValue metricValue;
	public:
		uint64_t kernelDuration;
		cuptiObj();
		int init();
		int registerCallbacks();
		long runKernelTime();
		int setup(const char*);
		int unsetup();
		int getNumPass();
		void flushAll();
		void startNewPass(int);
		bool checkAfterPass();
		CUpti_MetricValue metricGenerateValue();
		CUpti_MetricID getMetricId();
		CUpti_MetricID getMetricValue();
	
};

cuptiObj::cuptiObj()
{
	context = 0;
	device = 0;
}

int cuptiObj::init()
{  
	int computeCapabilityMajor=0;
	int computeCapabilityMinor=0;
	int deviceNum = 0;
	int deviceCount;
	char deviceName[32];

	// make sure activity is enabled before any CUDA API
	CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

	DRIVER_API_CALL(cuInit(0));
	DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0) {
		printf("There is no device supporting CUDA.\n");
		return -2;
	}

	DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
	DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));
	printf("CUDA Device Name: %s\n", deviceName);

	DRIVER_API_CALL(cuDeviceComputeCapability(&computeCapabilityMajor,
											&computeCapabilityMinor,
											device));
	DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
	
	return 0;
}

int cuptiObj::registerCallbacks()
{
	CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
	return 0;
}

int cuptiObj::setup(const char *metricName)
{
	// setup launch callback for event collection
	CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getMetricValueCallback, &metricData));
	CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
								 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));

	// allocate space to hold all the events needed for the metric
	if (cuptiMetricGetIdFromName(device, metricName, &metricId) != CUPTI_SUCCESS)
	{
		CUPTI_CALL(cuptiUnsubscribe(subscriber));
		return -1;
	}
	//printf("metricId: %d, metricName: %s\n", metricId, metricName);

	CUPTI_CALL(cuptiMetricGetNumEvents(metricId, &metricData.numEvents));
	metricData.device = device;
	metricData.eventIdArray = (CUpti_EventID *)malloc(metricData.numEvents * sizeof(CUpti_EventID));
	metricData.eventValueArray = (uint64_t *)malloc(metricData.numEvents * sizeof(uint64_t));
	metricData.eventIdx = 0;
	return 0;
}

int cuptiObj::unsetup()
{
	CUPTI_CALL(cuptiUnsubscribe(subscriber));
	return 0;
}

int cuptiObj::getNumPass()
{
	CUPTI_CALL(cuptiMetricCreateEventGroupSets(context, sizeof(metricId), &metricId, &passData));
	//printf("running %d passes for calculating metric. number of required events: %d\n\n", passData->numSets, metricData.numEvents);
	if (passData != 0)
		return passData->numSets;
	printf("getNumPass() -> metricId: %d error in cuptiMetricCreateEventGroupSets\n", metricId);
	return 0;
}

void cuptiObj::flushAll()
{
	CUPTI_CALL(cuptiActivityFlushAll(0));
	kernelDuration = __kernelDuration;
}

void cuptiObj::startNewPass(int pass)
{
	metricData.eventGroups = passData->sets + pass;
}

bool cuptiObj::checkAfterPass()
{
	if (metricData.eventIdx != metricData.numEvents) {
		fprintf(stderr, "error: expected %u metric events, got %u\n",
				metricData.numEvents, metricData.eventIdx);
		return -1;
	}
	return 0;
}

CUpti_MetricValue cuptiObj::metricGenerateValue()
{
	CUPTI_CALL(cuptiMetricGetValue(device, metricId,
								 metricData.numEvents * sizeof(CUpti_EventID),
								 metricData.eventIdArray,
								 metricData.numEvents * sizeof(uint64_t),
								 metricData.eventValueArray,
								 kernelDuration, &metricValue));
	return metricValue;
}

CUpti_MetricID cuptiObj::getMetricId()
{
	return metricId;
}