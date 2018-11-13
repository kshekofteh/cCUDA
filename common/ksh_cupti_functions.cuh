#include <ksh_cupti_api.cuh>

int ksh_cupti_init(int argc, char *argv[], CUcontext &context, CUdevice &device)
{  
	int computeCapabilityMajor=0;
	int computeCapabilityMinor=0;
	int deviceNum;
	int deviceCount;
	char deviceName[32];

	printf("Usage: %s [device_num] [metric_name]\n", argv[0]);

	// make sure activity is enabled before any CUDA API
	CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

	DRIVER_API_CALL(cuInit(0));
	DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0) {
		printf("There is no device supporting CUDA.\n");
		return -2;
	}

	if (argc > 1)
		deviceNum = atoi(argv[1]);
	else
		deviceNum = 0;
	printf("CUDA Device Number: %d\n", deviceNum);

	DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
	DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));
	printf("CUDA Device Name: %s\n", deviceName);

	DRIVER_API_CALL(cuDeviceComputeCapability(&computeCapabilityMajor,
											&computeCapabilityMinor,
											device));
	DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
	
	return 0;
}


template<class valueType> valueType getMetricValue(CUpti_MetricValueKind valueKind, CUpti_MetricID metricId, CUpti_MetricValue metricValue)
{
	size_t valueKindSize = sizeof(valueKind);
	CUPTI_CALL(cuptiMetricGetAttribute(metricId, CUPTI_METRIC_ATTR_VALUE_KIND,
									   &valueKindSize, &valueKind));
	printf("...\n");
	switch (valueKind) {
		case CUPTI_METRIC_VALUE_KIND_DOUBLE:
			//printf("Metric %s = %f\n", metricName, metricValue.metricValueDouble);
			return metricValue.metricValueDouble;
		case CUPTI_METRIC_VALUE_KIND_UINT64:
			//printf("Metric %s = %llu\n", metricName, (unsigned long long)metricValue.metricValueUint64);
			return (unsigned long long)metricValue.metricValueUint64;
		case CUPTI_METRIC_VALUE_KIND_INT64:
			//printf("Metric %s = %lld\n", metricName, (long long)metricValue.metricValueInt64);
			return (long long)metricValue.metricValueInt64;
		case CUPTI_METRIC_VALUE_KIND_PERCENT:
			//printf("Metric %s = %f%%\n", metricName, metricValue.metricValuePercent);
			return metricValue.metricValuePercent;
		case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
			//printf("Metric %s = %llu bytes/sec\n", metricName, (unsigned long long)metricValue.metricValueThroughput);
			return (unsigned long long)metricValue.metricValueThroughput;
		case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
			//printf("Metric %s = utilization level %u\n", metricName, (unsigned int)metricValue.metricValueUtilizationLevel);
			return (unsigned int)metricValue.metricValueUtilizationLevel;
		default:
			fprintf(stderr, "error: unknown value kind\n");
			//exit(-1);
	}
}