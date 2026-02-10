#pragma once
#include <cuda_runtime.h>

class CoarseGraindGPUTimer{
    private:
        cudaEvent_t startEvent, stopEvent;
        float elapsedTime;
    
    public:
        CoarseGraindGPUTimer();
        ~CoarseGraindGPUTimer();

        void start(cudaStream_t stream = 0); // default stream 0
        void stop(cudaStream_t stream = 0);
        float getElapsedTime(); // in milliseconds
};