// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <array>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "gtest/gtest.h"

#define EXPECT_CUDA_SUCCESS(code) EXPECT_EQ(cudaSuccess, code)
#define EXPECT_CUBLAS_SUCCESS(code) EXPECT_EQ(CUBLAS_STATUS_SUCCESS, code)

namespace {

class CublasTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cublasCreate(&cublasHandle);
    cudaMalloc(&devX, 3 * sizeof(float));
    cudaMalloc(&devY, 3 * sizeof(float));
    cudaMalloc(&devA, 3 * 3 * sizeof(float));
    cudaMalloc(&devB, 3 * 4 * sizeof(float));
    cudaMalloc(&devC, 3 * 4 * sizeof(float));
  }

  void TearDown() override {
    cublasDestroy(cublasHandle);
    cudaFree(devX);
    cudaFree(devY);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
  }

  cublasHandle_t cublasHandle;
  float *devX;
  float *devY;
  float *devA;
  float *devB;
  float *devC;
};

TEST_F(CublasTest, Level1Reductions) {
  std::array<float, 3> hostX = {{2, 3, 1}};
  int n = hostX.size();
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostX.data(), 1, devX, 1));

  int indexOfMax = 0;
  EXPECT_CUBLAS_SUCCESS(cublasIsamax(cublasHandle, n, devX, 1, &indexOfMax));
  EXPECT_EQ(2, indexOfMax);  // cublasIsamax uses 1-based indexing.

  int indexOfMin = 0;
  EXPECT_CUBLAS_SUCCESS(cublasIsamin(cublasHandle, n, devX, 1, &indexOfMin));
  EXPECT_EQ(3, indexOfMin);

  float sum = 0;
  EXPECT_CUBLAS_SUCCESS(cublasSasum(cublasHandle, n, devX, 1, &sum));
  EXPECT_EQ(6, sum);

  float norm2 = 0;
  EXPECT_CUBLAS_SUCCESS(cublasSnrm2(cublasHandle, n, devX, 1, &norm2));
  EXPECT_NEAR(14, norm2 * norm2, 1e-3);
}

TEST_F(CublasTest, Level1Scale) {
  float alpha = 2.0f;
  std::array<float, 3> hostX = {{2, 3, 1}};
  std::array<float, 3> expected = {{4, 6, 2}};
  int n = hostX.size();
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostX.data(), 1, devX, 1));
  EXPECT_CUBLAS_SUCCESS(cublasSscal(cublasHandle, n, &alpha, devX, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasGetVector(n, sizeof(float), devX, 1, hostX.data(), 1));
  EXPECT_EQ(expected, hostX);
}

TEST_F(CublasTest, Level1Transform) {
  std::array<float, 3> hostX = {{2, 3, 1}};
  std::array<float, 3> hostY = {{5, 4, 6}};
  std::array<float, 3> resultX;
  std::array<float, 3> resultY;
  int n = hostX.size();

  // saxpy
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostX.data(), 1, devX, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostY.data(), 1, devY, 1));
  float alpha = 2.0f;
  EXPECT_CUBLAS_SUCCESS(cublasSaxpy(cublasHandle, n, &alpha, devX, 1, devY, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasGetVector(n, sizeof(float), devY, 1, resultY.data(), 1));
  std::array<float, 3> saxpyExpected;
  for (int i = 0; i < 3; ++i) {
    saxpyExpected[i] = alpha * hostX[i] + hostY[i];
  }
  EXPECT_EQ(saxpyExpected, resultY);

  // copy
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostX.data(), 1, devX, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostY.data(), 1, devY, 1));
  EXPECT_CUBLAS_SUCCESS(cublasScopy(cublasHandle, n, devX, 1, devY, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasGetVector(n, sizeof(float), devY, 1, resultY.data(), 1));
  EXPECT_EQ(hostX, resultY);

  // dot
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostX.data(), 1, devX, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostY.data(), 1, devY, 1));
  float dotResult = 0;
  EXPECT_CUBLAS_SUCCESS(
      cublasSdot(cublasHandle, n, devX, 1, devY, 1, &dotResult));
  float expectedDotResult = 0;
  for (int i = 0; i < 3; ++i) {
    expectedDotResult += hostX[i] * hostY[i];
  }
  EXPECT_EQ(expectedDotResult, dotResult);

  // swap
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostX.data(), 1, devX, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostY.data(), 1, devY, 1));
  EXPECT_CUBLAS_SUCCESS(cublasSswap(cublasHandle, n, devX, 1, devY, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasGetVector(n, sizeof(float), devX, 1, resultX.data(), 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasGetVector(n, sizeof(float), devY, 1, resultY.data(), 1));
  EXPECT_EQ(resultX, hostY);
  EXPECT_EQ(resultY, hostX);

  // rot
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostX.data(), 1, devX, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostY.data(), 1, devY, 1));
  float c = 2;
  float s = 3;
  EXPECT_CUBLAS_SUCCESS(cublasSrot(cublasHandle, n, devX, 1, devY, 1, &c, &s));
  EXPECT_CUBLAS_SUCCESS(
      cublasGetVector(n, sizeof(float), devX, 1, resultX.data(), 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasGetVector(n, sizeof(float), devY, 1, resultY.data(), 1));
  std::array<float, 3> expectedRotX;
  std::array<float, 3> expectedRotY;
  for (int i = 0; i < 3; ++i) {
    expectedRotX[i] = c * hostX[i] + s * hostY[i];
    expectedRotY[i] = c * hostY[i] - s * hostX[i];
  }
  EXPECT_EQ(expectedRotX, resultX);
  EXPECT_EQ(expectedRotY, resultY);
}

TEST_F(CublasTest, Level2) {
  std::array<float, 3> hostX = {{1, 2, 3}};
  std::array<float, 3> hostY = {{6, 5, 4}};
  std::array<float, 3> resultY;
  // Note that A is in column-major order, so the rows here are really the rows
  // of the transpose.
  std::array<float, 3 * 3> hostA = {{0, 2, 1,  //
                                     2, 3, 4,  //
                                     0, 1, 0}};
  std::array<float, 3 * 3> resultA;
  int n = hostX.size();
  int m = hostA.size() / n;

  float alpha = 2;
  float beta = 3;

  // Gemv
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostX.data(), 1, devX, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostY.data(), 1, devY, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasSetMatrix(m, n, sizeof(float), hostA.data(), m, devA, m));

  cublasOperation_t trans = CUBLAS_OP_T;  // Transpose.
  EXPECT_CUBLAS_SUCCESS(cublasSgemv(cublasHandle, trans, m, n, &alpha, devA, m,
                                    devX, 1, &beta, devY, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasGetVector(n, sizeof(float), devY, 1, resultY.data(), 1));
  // Check against by-hand computation:
  //
  // A_transpose * x = {7, 20, 2}, then multiplying by alpha and adding beta * y
  // gives 2 * {7, 20, 2} + 3 * {6, 5, 4} = {32, 55, 16}
  std::array<float, 3> expectedGemv = {{32, 55, 16}};
  EXPECT_EQ(expectedGemv, resultY);

  // Ger
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostX.data(), 1, devX, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasSetVector(n, sizeof(float), hostY.data(), 1, devY, 1));
  EXPECT_CUBLAS_SUCCESS(
      cublasSetMatrix(m, n, sizeof(float), hostA.data(), m, devA, m));

  EXPECT_CUBLAS_SUCCESS(
      cublasSger(cublasHandle, m, n, &alpha, devX, 1, devY, 1, devA, m));
  EXPECT_CUBLAS_SUCCESS(
      cublasGetMatrix(m, n, sizeof(float), devA, m, resultA.data(), m));
  // Check against by-hand computation (expressing computations in column-major
  // order):
  //
  // first:
  // x * y^t = {1, 2, 3} * {6, 5, 4}^t = {{6, 12, 18}, {5, 10, 15}, {4, 8, 12}}
  //
  // then:
  // 2 * (x * y^t) = {{12, 24, 36}, {10, 20, 30}, {8, 16, 24}}
  //             A = {{ 0,  2,  1}, { 2,  3,  4}, {0,  1,  0}}
  // ---------------------------------------------------------
  //           sum = {{12, 26, 37}, {12, 23, 34}, {8, 17, 24}}
  std::array<float, 3 * 3> expectedGer = {{12, 26, 37, 12, 23, 34, 8, 17, 24}};
  EXPECT_EQ(expectedGer, resultA);
}

TEST_F(CublasTest, Gemm) {
  std::array<float, 3 * 3> hostA = {{0, 1, 2,  //
                                     3, 4, 5,  //
                                     6, 7, 8}};
  std::array<float, 3 * 4> hostB = {{0, 1, 2,  //
                                     3, 4, 5,  //
                                     6, 7, 8,  //
                                     9, 10, 11}};
  std::array<float, 3 * 4> hostC = {{1, 2, 1,  //
                                     4, 3, 0,  //
                                     1, 0, 2,  //
                                     4, 0, 1}};
  std::array<float, 3 * 4> resultC;
  int m = 3;
  int k = 3;
  int n = 4;

  float alpha = 2;
  float beta = 3;

  EXPECT_CUBLAS_SUCCESS(
      cublasSetMatrix(m, k, sizeof(float), hostA.data(), m, devA, m));
  EXPECT_CUBLAS_SUCCESS(
      cublasSetMatrix(k, n, sizeof(float), hostB.data(), k, devB, k));
  EXPECT_CUBLAS_SUCCESS(
      cublasSetMatrix(m, n, sizeof(float), hostC.data(), m, devC, m));

  cublasOperation_t opA = CUBLAS_OP_N;  // No transpose.
  cublasOperation_t opB = CUBLAS_OP_N;  // No transpose.
  EXPECT_CUBLAS_SUCCESS(cublasSgemm(cublasHandle, opA, opB, m, n, k, &alpha,
                                    devA, m, devB, k, &beta, devC, m));
  EXPECT_CUBLAS_SUCCESS(
      cublasGetMatrix(m, n, sizeof(float), devC, m, resultC.data(), m));

  // Check against by-hand computation (expressing computations in column-major
  // order):
  //
  // first:
  // A*B   = {{15, 18, 21}, {42,  54,  66}, { 69,  90, 111}, { 96, 126, 156}}
  //
  // then:
  // 2*A*B = {{30, 36, 42}, {84, 108, 132}, {138, 180, 222}, {192, 252, 312}}
  // 3*C   = {{ 3,  6,  3}, {12,   9,   0}, {  3,   0,   6}, { 12,   0,   3}}
  // ------------------------------------------------------------------------
  // sum   = {{33, 42, 45}, {96, 117, 132}, {141, 180, 228}, {204, 252, 315}}
  std::array<float, 3 * 4> expectedGemm = {{33, 42, 45,     //
                                            96, 117, 132,   //
                                            141, 180, 228,  //
                                            204, 252, 315}};
  EXPECT_EQ(expectedGemm, resultC);
}

}  // namespace
