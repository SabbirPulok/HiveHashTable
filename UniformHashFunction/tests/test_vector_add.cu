#include "vector_add.h"
#include "utils.h"
#include <gtest/gtest.h>

TEST(VectorAddTest, BasicTest) {
    const int N = 1024;
    float h_A[N], h_B[N], h_C[N], h_expected[N];

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
        h_expected[i] = h_A[i] + h_B[i];
    }

    // Perform vector addition
    vectorAdd(h_A, h_B, h_C, N);

    // Verify the result
    ASSERT_TRUE(compareVectors(h_C, h_expected, N));
}