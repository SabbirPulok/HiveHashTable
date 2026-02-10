#include "utils.h"
#include <cstdlib>
#include <cmath>

// Initialize a vector with random values
void initVector(float* vec, int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = static_cast<float>(rand()) / RAND_MAX; // Random value between 0 and 1
    }
}

// Compare two vectors for equality
bool compareVectors(const float* A, const float* B, const float* C, int N) {
    for (int i = 0; i < N; i++) {
        if (fabs(A[i] + B[i] - C[i]) > 1e-5) { // Allow for small floating-point differences
            printf("Index: %d, Expect: %f, Output: %f\n",i, A[i]+B[i], C[i]);
            return false;
        }
    }
    return true;
}