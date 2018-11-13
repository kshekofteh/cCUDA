#include <ksh_cupti_api.cuh>

#pragma once
// User data for event collection callback
typedef struct MetricData_st {
  // the device where metric is being collected
  CUdevice device;
  // the set of event groups to collect for a pass
  CUpti_EventGroupSet *eventGroups;
  // the current number of events collected in eventIdArray and
  // eventValueArray
  uint32_t eventIdx;
  // the number of entries in eventIdArray and eventValueArray
  uint32_t numEvents;
  // array of event ids
  CUpti_EventID *eventIdArray;
  // array of event values
  uint64_t *eventValueArray;
} MetricData_t;

static uint64_t __kernelDuration;


void CUPTIAPI
getMetricValueCallback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
  MetricData_t *metricData = (MetricData_t*)userdata;
  unsigned int i, j, k;

  // This callback is enabled only for launch so we shouldn't see
  // anything else.
  if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
    printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
    exit(-1);
  }

  // on entry, enable all the event groups being collected this pass,
  // for metrics we collect for all instances of the event
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cudaDeviceSynchronize();

    CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
                                           CUPTI_EVENT_COLLECTION_MODE_KERNEL));

    for (i = 0; i < metricData->eventGroups->numEventGroups; i++) {
      uint32_t all = 1;
      CUPTI_CALL(cuptiEventGroupSetAttribute(metricData->eventGroups->eventGroups[i],
                                             CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                             sizeof(all), &all));
      CUPTI_CALL(cuptiEventGroupEnable(metricData->eventGroups->eventGroups[i]));
    }
  }

  // on exit, read and record event values
  if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    cudaDeviceSynchronize();

    // for each group, read the event values from the group and record
    // in metricData
    for (i = 0; i < metricData->eventGroups->numEventGroups; i++) {
      CUpti_EventGroup group = metricData->eventGroups->eventGroups[i];
      CUpti_EventDomainID groupDomain;
      uint32_t numEvents, numInstances, numTotalInstances;
      CUpti_EventID *eventIds;
      size_t groupDomainSize = sizeof(groupDomain);
      size_t numEventsSize = sizeof(numEvents);
      size_t numInstancesSize = sizeof(numInstances);
      size_t numTotalInstancesSize = sizeof(numTotalInstances);
      uint64_t *values, normalized, sum;
      size_t valuesSize, eventIdsSize;

      CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                             CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                                             &groupDomainSize, &groupDomain));
      CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(metricData->device, groupDomain,
                                                    CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                                                    &numTotalInstancesSize, &numTotalInstances));
      CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                             CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                             &numInstancesSize, &numInstances));
      CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                             CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                             &numEventsSize, &numEvents));
      eventIdsSize = numEvents * sizeof(CUpti_EventID);
      eventIds = (CUpti_EventID *)malloc(eventIdsSize);
      CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                             CUPTI_EVENT_GROUP_ATTR_EVENTS,
                                             &eventIdsSize, eventIds));

      valuesSize = sizeof(uint64_t) * numInstances;
      values = (uint64_t *)malloc(valuesSize);

      for (j = 0; j < numEvents; j++) {
        CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                                            eventIds[j], &valuesSize, values));
        if (metricData->eventIdx >= metricData->numEvents) {
          fprintf(stderr, "error: too many events collected, metric expects only %d\n",
                  (int)metricData->numEvents);
          exit(-1);
        }

        // sum collect event values from all instances
        sum = 0;
        for (k = 0; k < numInstances; k++)
          sum += values[k];

        // normalize the event value to represent the total number of
        // domain instances on the device
        normalized = (sum * numTotalInstances) / numInstances;

        metricData->eventIdArray[metricData->eventIdx] = eventIds[j];
        metricData->eventValueArray[metricData->eventIdx] = normalized;
        metricData->eventIdx++;

        // print collected value
        {
          char eventName[128];
          size_t eventNameSize = sizeof(eventName) - 1;
          CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME,
                                            &eventNameSize, eventName));
          eventName[127] = '\0';
		  /*
          printf("\tEvent id:%ud, Event name:%s = %llu (", eventIds[j], eventName, (unsigned long long)sum);
          if (numInstances > 1) {
            for (k = 0; k < numInstances; k++) {
              if (k != 0)
                printf(", ");
              printf("%llu", (unsigned long long)values[k]);
            }
          }

          printf(")\n");
		  printf("\t%s (normalized) (%llu * %u) / %u = %llu\n",
                 eventName, (unsigned long long)sum,
                 numTotalInstances, numInstances,
                 (unsigned long long)normalized);
		 */
        }
      }

      free(values);
    }

    for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
      CUPTI_CALL(cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]));
  }
}


static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *rawBuffer;

  *size = 16 * 1024;
  rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = 0;

  if (*buffer == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUpti_Activity *record = NULL;
  CUpti_ActivityKernel3 *kernel;

  //since we launched only 1 kernel, we should have only 1 kernel record
  CUPTI_CALL(cuptiActivityGetNextRecord(buffer, validSize, &record));

  kernel = (CUpti_ActivityKernel3 *)record;
  if (kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL) {
    fprintf(stderr, "Error: expected kernel activity record, got %d\n", (int)kernel->kind);
    exit(-1);
  }

  __kernelDuration = kernel->end - kernel->start; 
  free(buffer);
}
