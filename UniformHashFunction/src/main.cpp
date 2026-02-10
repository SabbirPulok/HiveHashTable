#include<iostream>
using std::cout;

#include "vector_add.h"
#include "utils.h"
#include "experiment_kernel.h"

struct sExpConfig{
    size_t nbuckets;
    size_t nkeys;
};

sExpConfig Configs[]{
    // {64,64},
    //{10,100},
    {64,128},
    {128,256},
    {256,512},
    {1024,1024},
    //  {1024,2048},
    // {1024,4096},
    // {1024,8192},
    // {512*512, 256*256},
    // {512*512, 512*512},
    // {512*512, 1024*1024},
    // {512*512, 2048*2048}
};

int main()
{
    for(const auto &[nbuckets, nkeys]:Configs)
    {
        int nTry=10;
        cout<<"Experiment Configuration (nBuckets, nKeys): {"<<nbuckets<<","<<nkeys<<"}\n";
        LaunchExperimentWithRetries(nbuckets, nkeys, nTry);
    }
    // const int N = 1<<20;
    // size_t size = N * sizeof(float);

    // //allocate host memory
    // float *hA = (float*)malloc(size);
    // float *hB = (float*)malloc(size);
    // float *hC = (float*) malloc(size);

    // initVector(hA, N);
    // initVector(hB, N);

    // vectorAdd(hA, hB, hC, N);

    // bool success = compareVectors(hA, hB, hC, N); //Expected result: hC = hA + hB

    // if(success)
    //     cout<<"Vector Addition successful"<<std::endl;
    // else
    //     std::cerr<<"Vector Addiiton failed!"<<std::endl;

    // free(hA);
    // free(hB);
    // free(hC);

    return 0;
}
