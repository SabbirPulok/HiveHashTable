#pragma once
#include <cuda_runtime.h>

class CoarseGrainedGPUTimer{
    private:
        cudaEvent_t startEvent, stopEvent;
        float elapsedTime;
    
    public:
        CoarseGrainedGPUTimer();
        ~CoarseGrainedGPUTimer();

        void start(cudaStream_t stream = 0); // default stream 0
        void stop(cudaStream_t stream = 0);
        float getElapsedTime(); // in milliseconds
};