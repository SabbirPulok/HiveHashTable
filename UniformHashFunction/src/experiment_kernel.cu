#include "experiment_kernel.h"

//__device__ uint32_t (*noLookUpHash[]) (uint32_t) {hash1, hash2, murmurhash, cityhash};

//__device__ uint32_t (*hash) (uint32_t) = hash1;

__constant__ uint8_t spinwheel_table_const[256*256];

std::string decodeFunc[]{
    "crc64", 
    "crc32", 
    "hash1", 
    "hash2", 
    "murmurhash3", 
    "cityhash32", 
    "spinwheel_hash", 
    "spinwheel_hash8b", 
    "identity_hash"
    };
std::string decodeKeySequence[]{"random", "sequential"};

double expectedEmptyBuckets(const size_t nbuckets, const size_t nkeys)
{
    return (double)nbuckets * (std::pow((1.0 - (1.0/(double)nbuckets)),(double)nkeys));
}

double prCollision(const size_t nbuckets, const size_t nkeys)
{
  return 1.0 - std::pow((((double)nbuckets-1.0)/(double)nbuckets),(double)nkeys-1.0);
}

double ExpectedCollisions(const size_t nbuckets, const size_t nkeys)
{
    double m = (double)nbuckets;
    double n = (double)nkeys;
    
    //r  = (m-1)/m
    //ec = n - ((1-r^n)/(1-r))

    double r = (m-1.0)/m;
    return n - ((1.0-std::pow(r,n))/(1.0-r));
}
long double factorial(const long double n)
{
    return std::tgamma(n+1.0);
}

long double nChoosek(long double n, long double k)
{
    //n!/(n-k)!k!
    return std::tgamma(n+1.0L) / (std::tgamma(k+1.0L)*std::tgamma(n-k+1.0L));
}

//optimize for large factorial
long double ln_nChoosek(long double n, long double k)
{
    //n!/(n-k)!k!
   return log(factorial(n)) - log(factorial(k)) - log(factorial(n-k));
}

long double ExpectedNumberOfBucketsWithKelements_A(const long double B, const long double n, const long double k)
{
    double oneOverB { 1.0L/B };
    double missProb { 1.0L - oneOverB};

    return exp(log(B) + ln_nChoosek(n,k) + k*log(oneOverB) + (n-k) * log(missProb));
}

long double ExpectedNumberOfBucketsWithKelements_B(const long double B, const long double n, const long double k)
{
    double oneOverB { 1.0L/B };
    double missProb { 1.0L - oneOverB};

    return B* nChoosek(n,k)* std::pow(oneOverB, k)* std::pow(missProb, n-k);
}

long double TheoreticalProbability(const long double B, const long double n, const long double k)
{
    double oneOverB { 1.0L/B };
    double missProb { 1.0L - oneOverB};

    return std::pow(oneOverB, k) * std::pow(missProb, n-k);
}

void StatsMeasurement(vector<uint64_t>buckets, AllStats *stats, int index, double nKeys, double &avgDeviationFromMean, vector<double>&observed_k)
{    
    // long double theoretical_probability_ln = ExpectedNumberOfBucketsWithKelements_A(10.0L,20.0L, 2.0);
    // long double theoretical_probability_wln = ExpectedNumberOfBucketsWithKelements_B(10.0L,20.0L, 2.0);

    // std::cout<<"Expected(withLn): "<<theoretical_probability_ln<<"\nExpected(woLn): "<<theoretical_probability_wln<<std::endl;
    // std::exit(1);

    int emptyBucketCount = 0;
    tStats sumBuckets;
    int nCollisions = 0;

    for(auto &b: buckets)
    {
        // stats->bucketCountStats[index] += b;

        if(b==0)
            emptyBucketCount++;
        
        if(b>1)
            nCollisions += (b-1);
        // stats->bucketCollisionsStats[index] += (b<=1)?0.0:(b-1.0);
        sumBuckets += b;
    }

    stats->bucketCountStatsMin[index] += sumBuckets.Min;
    stats->bucketCountStatsMax[index] += sumBuckets.Max;
    stats->emptyBucketStats[index] += emptyBucketCount;
    stats->bucketCollisionsStats[index] += nCollisions;
    
    // For each K
    //     match=0
    //     For each B
    //         if(k==b)
    //             match++
    //     observation= number of buckets with k elements = match    
    //     observed[k] += observation

    for(double k=0.0; k<=nKeys; ++k)
    {
        double nMatchingBuckets = 0.0;

        for(auto &b: buckets)
        {
            if(b==k)
                ++nMatchingBuckets;
        }

        observed_k[k] += nMatchingBuckets;
        //cout<<"k: "<<k<<" nMathcingbuckets: "<<nMatchingBuckets<<std::endl;


        // double observed_probability = nMatchingBuckets / nKeys;
        // double theoretical_probability = TheoreticalProbability2((double)buckets.size(), nKeys, k);
        // double diff = theoretical_probability - observed_probability;

        // avgDeviationFromMean+=diff;

        // cout<<"K: "<<k<<", Observed Prob: "<<std::fixed<<std::setprecision(10)<<observed_probability<< 
        // ", Theoretical Probability: "<<theoretical_probability<<", Difference: "<<diff<<std::endl;

    }

    //cout<<"Empty Bucket Count ["<<index<<"] : "<<emptyBucketCount<<std::endl;
}

void AddStatsToCSV(
    tTabularResults file,
    AllStats *stats,
    size_t nbuckets, 
    size_t nkeys,
    int keySequence,
    vector<double>rss
    )
{
    file.BeginTable();

    for(int i=0; i<nFunctions; ++i)
    {
        file.BeginRow();
        file.AddColumn("Timestamp",TimestampLocal());
        file.AddColumn("Hash Function", decodeFunc[i]);
        file.AddColumn("nBuckets",nbuckets);
        file.AddColumn("nKeys", nkeys);
        file.AddColumn("KeySequence",decodeKeySequence[keySequence]);
        file.AddColumn("RSS", rss[i]);
        file.AddColumn("Expected # of Empty Buckets",expectedEmptyBuckets(nbuckets,nkeys));
        file.AddColumn("Empty Bucket Stats",stats->emptyBucketStats[i]);
        file.AddColumn("Expected # of collisions",ExpectedCollisions(nbuckets,nkeys));
        file.AddColumn("Probability of collisions",prCollision(nbuckets,nkeys));
        file.AddColumn("Bucket Collision Stats",stats->bucketCollisionsStats[i]);
        file.AddColumn("Bucket Count Stats Min",stats->bucketCountStatsMin[i]);
        file.AddColumn("Bucket Count Stats Max",stats->bucketCountStatsMax[i]);
        file.AddColumn("Elpased Time Stats", stats->elpTimeStats[i]);        
        file.EndRow();
    }
    file.EndTable();
    std::ofstream csvfile("Hash_Function_Study_Chapter_GPU_"+decodeKeySequence[keySequence]+std::to_string(nbuckets)+"b*"+std::to_string(nkeys)+"k.csv");
    file.CreateCSV(csvfile);
    csvfile.close();
}

void LaunchExperimentWithRetries(size_t nbuckets, size_t nkeys, int nTry)
{
    vector<uint64_t> CRC64_Table;
    vector<uint32_t> CRC32_Table;
    vector<uint16_t> SpinWheelTable;
    vector<uint8_t> SpinWheelTable8b;    

    generate_CRC64_table(CRC64_Table);
    generate_CRC32_table(CRC32_Table);
    generate_SpinWheel_table(SpinWheelTable,PRIME_N);
    generate_SpinWheel_table_8b(SpinWheelTable8b,PRIME_N);

    size_t sizeCRC64 = CRC64_Table.size() * sizeof(uint64_t);
    size_t sizeCRC32 = CRC32_Table.size() * sizeof(uint32_t);
    size_t sizeSpinWheel = SpinWheelTable.size() * sizeof(uint16_t);
    size_t sizeSpinWheel8b = SpinWheelTable.size() * sizeof(uint8_t);


    uint64_t *d_CRC64_Table;
    uint32_t *d_CRC32_Table;
    uint16_t *d_SpinWheelTable;
    uint8_t *d_SpinWheelTable8b;

    cudaMalloc((void**)&d_CRC64_Table, sizeCRC64);
    cudaMalloc((void**)&d_CRC32_Table, sizeCRC32);
    cudaMalloc((void**)&d_SpinWheelTable, sizeSpinWheel);
    cudaMalloc((void**)&d_SpinWheelTable8b, sizeSpinWheel8b);

    cudaMemcpy(d_CRC64_Table, CRC64_Table.data(), sizeCRC64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_CRC32_Table, CRC32_Table.data(), sizeCRC32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_SpinWheelTable, SpinWheelTable.data(), sizeSpinWheel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_SpinWheelTable8b, SpinWheelTable8b.data(), sizeSpinWheel8b, cudaMemcpyHostToDevice);


    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(spinwheel_table_const, SpinWheelTable8b.data(), sizeSpinWheel8b, 0, cudaMemcpyHostToDevice));

    vector<uint64_t>h_buckets;
    h_buckets.resize(nbuckets);
    
    std::fill(h_buckets.begin(), h_buckets.end(), 0ULL);

    uint64_t *d_buckets;
    cudaMalloc((void**)&d_buckets, nbuckets* sizeof(uint64_t));

    CHECK_CUDA_ERROR(cudaMemcpy(d_buckets, h_buckets.data(), nbuckets*sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    // for(keySequences)
    //     for (each config)
    //         // for each hash function
    //             observed[nFunctions][K]
    //             for(each retry)
    //                 for each hash function
    //                     call hash function kernel
    //                     Note: At this point hashing to buckets is complete
    //                     We need to gather experimental results
    //                     For each K
    //                         match=0
    //                         For each B
    //                             if(k==b)
    //                                 match++
    //                         observation= number of buckets with k elements = match    
    //                         observed[func][k] += observation
                
    //             deviation[nFunc]{0}
                
    //             for each func
    //                 for each k
    //                    deviation[func]  += observed[func][k]/nTry - expected(|B|, N, k)
                

    //             deviation[nFunctions]
    //             deviation[hash_function]{0}

    //             for each K(0~N)
    //                 deviation  += observed[k] - expected(|B|, N, k)

                
    for(int sequence=erandom; sequence<nKeySequence; ++sequence)
    {
        sequence==erandom?std::cout<<"Random..\n":std::cout<<"Sequential...\n";

        tTabularResults results;
        AllStats *statesSummary = new AllStats();

        static uint32_t lastSequentialKey {0};
        
        //observed array
        vector<vector<double>>func_observed_k;
        func_observed_k.resize(nFunctions);
        
        for(int f=0;f<nFunctions; f++)
        {
            func_observed_k[f].resize(nkeys+1);
            for(int k=0; k<=nkeys;k++)
            {
                func_observed_k[f][k] = 0.0;
            }
        }
        

        //Retry loop
        for(int i=0; i<nTry; i++)
        {
            double avgDeviationFromMean = 0.0;

            //keys generation
            vector<uint32_t>h_keys;
            h_keys.resize(nkeys);

            std::random_device rnd;
            std::mt19937 gen(rnd());
            
            // std::uniform_int_distribution<uint32_t> keyDist(0u, UINT32_MAX);

            for(auto &key: h_keys)
            {
                key = ((sequence==erandom)?gen():lastSequentialKey++);
            }

            //pass keys
            uint32_t *d_keys;
            cudaMalloc((void**)&d_keys, nkeys*sizeof(uint32_t));

            CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), nkeys*sizeof(uint32_t), cudaMemcpyHostToDevice));

            LaunchExperiment(d_keys, h_buckets, d_buckets, statesSummary, d_CRC64_Table, d_CRC32_Table, d_SpinWheelTable, d_SpinWheelTable8b, 
            nbuckets, nkeys, avgDeviationFromMean, func_observed_k);
            
            //cout<<"---------Try: "<<i<<"--------------\n";
            //cout<<"Average Deviation from Mean: "<<avgDeviationFromMean<<std::endl;
        }

//      deviation[func]  += observed[func][k]/nTry - expected(|B|, N, k)

        vector<double>deviation(nFunctions,0.0);
        vector<double>rss(nFunctions,0.0); //residual sum of squares
        
        

        for(int func{0}; func<nFunctions; func++)
        {
            //deviation[func]=0.0;
            double sum_observed_k = 0.0;
            double sum_theoretical_k = 0.0;

            for (int k=0; k<=nkeys; k++) 
            {
                sum_observed_k+=(func_observed_k[func][k])/nTry;
                sum_theoretical_k+=ExpectedNumberOfBucketsWithKelements_A(nbuckets,nkeys,k);
                
                double diff = (func_observed_k[func][k] / (nTry)) - ExpectedNumberOfBucketsWithKelements_A(nbuckets,nkeys,k);
                deviation[func] += diff;
                rss[func] += diff*diff;

                //std::cout<<"k: "<<k<<", Observed: "<<func_observed_k[func][k]/nTry<<", Expected: "<<ExpectedNumberOfBucketsWithKelements_B(nbuckets,nkeys,k)<<std::endl;
            }
            // std::cout<<"Sum Observed K: "<<sum_observed_k<<std::endl;
            // std::cout<<"Sum Theoretical K: "<<sum_theoretical_k<<std::endl;
            // std::cout<<"Deviation for "<<decodeFunc[func]<<" : "<<deviation[func]<<std::endl;
            std::cout<<"RSS for "<<decodeFunc[func]<<" : "<<rss[func]<<std::endl;

        }

        AddStatsToCSV(results, statesSummary, nbuckets, nkeys, sequence, rss);   
    }    

    cout<<"Result saved on CSV files...\n";
}
void LaunchExperiment(
    uint32_t* keys,
    vector<uint64_t>h_buckets, 
    uint64_t* d_buckets, 
    AllStats *stats, 
    uint64_t *CRC64_Table, 
    uint32_t *CRC32_Table, 
    uint16_t *SpinWheelTable,
    uint8_t *SpinWheelTable8b, 
    size_t nbuckets, 
    size_t nkeys,
    double &avgDeviationFromMean,
    vector<vector<double>>&func_observed_k
    )
{
    int threadsPerBlock = 1024;
    int blocksPerGrid = (nkeys + threadsPerBlock -1)/threadsPerBlock;


    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   
    vector<double>observed_k;
    observed_k.resize(nkeys+1);

    //CRC64
    cudaEventRecord(start, 0);    
    
    crc64Kernel<<<blocksPerGrid,threadsPerBlock>>>(d_buckets, keys, nbuckets, nkeys, CRC64_Table);

    CHECK_CUDA_ERROR(cudaGetLastError());

    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();    
    
    cudaEventElapsedTime(&elapsedTime, start, stop);

    stats->elpTimeStats[ecrc64] += elapsedTime;
    

    cudaMemcpy(h_buckets.data(), d_buckets, nbuckets*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    observed_k = func_observed_k[ecrc64];
    // cout<<"\nFunc_observed_k_size: "<<func_observed_k[ecrc64].size()<<", Observed_k: "<<std::all_of(observed_k.begin(), observed_k.end(), [](int i){
    //     return i==0;
    // });
    // cout<<"\n";
    StatsMeasurement(h_buckets, stats, ecrc64, (double)nkeys, avgDeviationFromMean, observed_k);
    
    func_observed_k[ecrc64] = observed_k;

    //reset buckets
    //cudaMemset works byte wise; sets each byte of buckets to 0x00,
    // not 0ULL, then memset will be overflown and only take least significant bits
    cudaMemset(d_buckets, 0, nbuckets * sizeof(uint64_t));

    
    //CRC32
    elapsedTime = 0.0;

    cudaEventRecord(start, 0);    
    
    crc32Kernel<<<blocksPerGrid,threadsPerBlock>>>(d_buckets, keys, nbuckets, nkeys, CRC32_Table);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();    
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    stats->elpTimeStats[ecrc32] += elapsedTime;

    cudaMemcpy(h_buckets.data(), d_buckets, nbuckets*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    
    observed_k = func_observed_k[ecrc32];
    // cout<<"\nFunc_observed_k_size: "<<func_observed_k[ecrc32].size()<<", Observed_k: "<<std::all_of(observed_k.begin(), observed_k.end(), [](int i){
    //     return i==0;
    // });
    // cout<<"\n";

    StatsMeasurement(h_buckets, stats, ecrc32, (double)nkeys, avgDeviationFromMean, observed_k);
    func_observed_k[ecrc32] = observed_k;
        

    // elapsedTime = 0.0;

    // cudaEventRecord(start, 0);    
    
    // spinWheelKernel<<<blocksPerGrid,threadsPerBlock>>>(d_buckets, keys, nbuckets, nkeys, SpinWheelTable);
    // CHECK_CUDA_ERROR(cudaGetLastError());
    
    // cudaEventRecord(stop,0);

    // cudaEventSynchronize(stop);
    // cudaDeviceSynchronize();    
    
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // stats->elpTimeStats[espinwheel_hash] += elapsedTime;

    // cudaMemcpy(h_buckets.data(), d_buckets, nbuckets*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // StatsMeasurement(h_buckets, stats, espinwheel_hash, (double)nkeys);

    cudaMemset(d_buckets, 0, nbuckets * sizeof(uint64_t));

    //SpinWheel8b
    elapsedTime = 0.0;

    cudaEventRecord(start, 0);    
    
    spinWheelKernel8b<<<blocksPerGrid,threadsPerBlock>>>(d_buckets, keys, nbuckets, nkeys, SpinWheelTable8b);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();    
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    stats->elpTimeStats[espinwheel_hash8b] += elapsedTime;

    cudaMemcpy(h_buckets.data(), d_buckets, nbuckets*sizeof(uint64_t), cudaMemcpyDeviceToHost);


    observed_k = func_observed_k[espinwheel_hash8b];

    StatsMeasurement(h_buckets, stats, espinwheel_hash8b, (double)nkeys, avgDeviationFromMean, observed_k);

    func_observed_k[espinwheel_hash8b] = observed_k;

    cudaMemset(d_buckets, 0, nbuckets * sizeof(uint64_t));

    //hash1
    elapsedTime = 0.0;

    cudaEventRecord(start, 0);    
    
    hash1Kernel<<<blocksPerGrid,threadsPerBlock>>>(d_buckets, keys, nbuckets, nkeys);

    CHECK_CUDA_ERROR(cudaGetLastError());
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();    
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    stats->elpTimeStats[ehash1] += elapsedTime;

    cudaMemcpy(h_buckets.data(), d_buckets, nbuckets*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    observed_k = func_observed_k[ehash1];
    StatsMeasurement(h_buckets, stats, ehash1, (double)nkeys, avgDeviationFromMean, observed_k);
    func_observed_k[ehash1] = observed_k;

    cudaMemset(d_buckets, 0, nbuckets * sizeof(uint64_t));

    //hash2
    elapsedTime = 0.0;

    cudaEventRecord(start, 0);    
    
    hash2Kernel<<<blocksPerGrid,threadsPerBlock>>>(d_buckets, keys, nbuckets, nkeys);

    CHECK_CUDA_ERROR(cudaGetLastError());
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();    
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    stats->elpTimeStats[ehash2] += elapsedTime;

    cudaMemcpy(h_buckets.data(), d_buckets, nbuckets*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    observed_k = func_observed_k[ehash2];
    StatsMeasurement(h_buckets, stats, ehash2, (double)nkeys, avgDeviationFromMean, observed_k);
    func_observed_k[ehash2] = observed_k;

    cudaMemset(d_buckets, 0, nbuckets * sizeof(uint64_t));

    //city
    elapsedTime = 0.0;

    cudaEventRecord(start, 0);    
    
    cityHashKernel<<<blocksPerGrid,threadsPerBlock>>>(d_buckets, keys, nbuckets, nkeys);

    CHECK_CUDA_ERROR(cudaGetLastError());
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();    
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    stats->elpTimeStats[ecityhash32] += elapsedTime;

    cudaMemcpy(h_buckets.data(), d_buckets, nbuckets*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    observed_k = func_observed_k[ecityhash32];
    StatsMeasurement(h_buckets, stats, ecityhash32, (double)nkeys, avgDeviationFromMean, observed_k);
    func_observed_k[ecityhash32] = observed_k;

    cudaMemset(d_buckets, 0, nbuckets * sizeof(uint64_t));

    //Murmur
    elapsedTime = 0.0;

    cudaEventRecord(start, 0);    
    
    murmurHashKernel<<<blocksPerGrid,threadsPerBlock>>>(d_buckets, keys, nbuckets, nkeys);

    CHECK_CUDA_ERROR(cudaGetLastError());
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();    
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    stats->elpTimeStats[emurmurhash3] += elapsedTime;

    cudaMemcpy(h_buckets.data(), d_buckets, nbuckets*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    observed_k = func_observed_k[emurmurhash3];

    StatsMeasurement(h_buckets, stats, emurmurhash3, (double)nkeys, avgDeviationFromMean, observed_k);

    func_observed_k[emurmurhash3] = observed_k;

    cudaMemset(d_buckets, 0, nbuckets * sizeof(uint64_t));
   
    //Identity Hash
    elapsedTime = 0.0;

    cudaEventRecord(start, 0);    
    
    identityHashKernel<<<blocksPerGrid,threadsPerBlock>>>(d_buckets, keys, nbuckets, nkeys);

    CHECK_CUDA_ERROR(cudaGetLastError());
    
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();    
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    stats->elpTimeStats[eidentityhash] += elapsedTime;

    cudaMemcpy(h_buckets.data(), d_buckets, nbuckets*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    observed_k = func_observed_k[eidentityhash];

    StatsMeasurement(h_buckets, stats, eidentityhash, (double)nkeys, avgDeviationFromMean, observed_k);

    func_observed_k[eidentityhash] = observed_k;

    cudaMemset(d_buckets, 0, nbuckets * sizeof(uint64_t));


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__global__ void identityHashKernel(
    uint64_t* buckets,
    uint32_t* keys,
    size_t nbuckets, 
    size_t nkeys)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=nkeys)
        return;
    
    uint32_t key = keys[tid];
    size_t index = identityhash(key) % nbuckets;
    atomicAdd((unsigned long long int*)&buckets[index],1ULL);

}

__global__ void crc64Kernel(
    uint64_t* buckets,
    uint32_t* keys,
    size_t nbuckets, 
    size_t nkeys, 
    uint64_t* CRC64_Table)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=nkeys)
        return;
    
    uint32_t key = keys[tid];
    size_t index = crc64(CRC64_Table, key) % nbuckets;
    atomicAdd((unsigned long long int*)&buckets[index],1ULL);

}

 __global__ void crc32Kernel(
    uint64_t* buckets, uint32_t* keys,
    size_t nbuckets,
    size_t nkeys, 
    uint32_t* CRC32_Table)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=nkeys)
        return;
    
    uint32_t key = keys[tid];
    size_t index = crc32(CRC32_Table, key) % nbuckets;
    atomicAdd((unsigned long long int*)&buckets[index],1ULL);
}

 __global__ void spinWheelKernel(
    uint64_t* buckets,
    uint32_t* keys, 
    size_t nbuckets, 
    size_t nkeys, 
    uint16_t* SpinWheel_Table)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=nkeys)
        return;
    
    uint32_t key = keys[tid];    
    size_t index = spinwheel(SpinWheel_Table, key) % nbuckets;
    atomicAdd((unsigned long long int*)&buckets[index],1ULL);
}

 __global__ void spinWheelKernel8b(
    uint64_t* buckets,
    uint32_t* keys, 
    size_t nbuckets, 
    size_t nkeys, 
    uint8_t* SpinWheel_Table8b)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=nkeys)
        return;
    
    uint32_t key = keys[tid];
    // 32 threads search for only 4 lookups on at most 4 cachlines (L1 cache hit)
    //size_t index = spinwheel8b(SpinWheel_Table8b, key) % nbuckets; 
    size_t index = spinwheel8b(spinwheel_table_const, key) % nbuckets; 
    atomicAdd((unsigned long long int*)&buckets[index],1ULL);
}

 __global__ void hash1Kernel(
    uint64_t* buckets,
    uint32_t* keys, 
    size_t nbuckets, 
    size_t nkeys)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=nkeys)
        return;
    
    uint32_t key = keys[tid];
    size_t index = hash1(key) % nbuckets;
    atomicAdd((unsigned long long int*)&buckets[index],1ULL);
} 

 __global__ void hash2Kernel(
    uint64_t* buckets,
    uint32_t* keys, 
    size_t nbuckets, 
    size_t nkeys)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=nkeys)
        return;
    
    uint32_t key = keys[tid];
    size_t index = hash2(key) % nbuckets;
    atomicAdd((unsigned long long int*)&buckets[index],1ULL);
} 

 __global__ void cityHashKernel(
    uint64_t* buckets, 
    uint32_t* keys, 
    size_t nbuckets, 
    size_t nkeys)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=nkeys)
        return;
    
    uint32_t key = keys[tid];
    size_t index = cityhash(key) % nbuckets;
    atomicAdd((unsigned long long int*)&buckets[index],1ULL);
} 

 __global__ void murmurHashKernel(
    uint64_t* buckets,
    uint32_t* keys, 
    size_t nbuckets, 
    size_t nkeys)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=nkeys)
        return;
    
    uint32_t key = keys[tid];
    size_t index = murmurhash(key) % nbuckets;
    atomicAdd((unsigned long long int*)&buckets[index],1ULL);
} 

