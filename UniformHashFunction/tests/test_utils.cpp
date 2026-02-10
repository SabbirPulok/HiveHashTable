#include "utils.h"
#include <gtest/gtest.h>

TEST(UtilsTest, VectorInitialization) {
    const int N = 1024;
    float h_vec[N];

    initializeVector(h_vec, N);

    for (int i = 0; i < N; i++) {
        EXPECT_GE(h_vec[i], 0.0f);
        EXPECT_LE(h_vec[i], 1.0f);
    }
}

TEST(UtilsTest, VectorComparison) {
    const int N = 1024;
    float h_A[N], h_B[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    EXPECT_TRUE(compareVectors(h_A, h_B, N));
}