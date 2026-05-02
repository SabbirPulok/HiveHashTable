#include "GPUTimer.h"

CoarseGrainedGPUTimer::CoarseGrainedGPUTimer()
{
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    elapsedTime = 0.0f;
}


CoarseGrainedGPUTimer::~CoarseGrainedGPUTimer()
{
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void CoarseGrainedGPUTimer::start(cudaStream_t stream) // default stream 0
{
    cudaEventRecord(startEvent, stream);
}

void CoarseGrainedGPUTimer::stop(cudaStream_t stream)
{
    cudaEventRecord(stopEvent, stream);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
}
float CoarseGrainedGPUTimer::getElapsedTime() // in milliseconds
{
    return elapsedTime;
}