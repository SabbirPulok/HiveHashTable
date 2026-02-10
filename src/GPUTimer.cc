#include "GPUTimer.h"

CoarseGraindGPUTimer::CoarseGraindGPUTimer()
{
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    elapsedTime = 0.0f;
}


CoarseGraindGPUTimer::~CoarseGraindGPUTimer()
{
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void CoarseGraindGPUTimer::start(cudaStream_t stream) // default stream 0
{
    cudaEventRecord(startEvent, stream);
}

void CoarseGraindGPUTimer::stop(cudaStream_t stream)
{
    cudaEventRecord(stopEvent, stream);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
}
float CoarseGraindGPUTimer::getElapsedTime() // in milliseconds
{
    return elapsedTime;
}