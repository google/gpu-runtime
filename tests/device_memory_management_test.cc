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

#include <cuda_runtime.h>

#include "tests/common.h"
#include "gtest/gtest.h"

namespace {

TEST(CudartDeviceMemoryManagement, BasicUsage) {
  int *src;
  int *dst;
  int *device;

  EXPECT_CUDA_SUCCESS(cudaMallocHost(&src, sizeof(int)));
  EXPECT_CUDA_SUCCESS(cudaMallocHost(&dst, sizeof(int)));
  EXPECT_CUDA_SUCCESS(cudaMalloc(&device, sizeof(int)));

  *src = 42;

  EXPECT_CUDA_SUCCESS(
      cudaMemcpy(device, src, sizeof(int), cudaMemcpyHostToDevice));
  EXPECT_CUDA_SUCCESS(
      cudaMemcpy(dst, device, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(42, *dst);

  EXPECT_CUDA_SUCCESS(cudaMemset(device, 100, sizeof(int)));
  EXPECT_CUDA_SUCCESS(
      cudaMemcpy(dst, device, sizeof(int), cudaMemcpyDeviceToHost));

  int expectedMemsetOutput = (((((100 << 8) + 100) << 8) + 100) << 8) + 100;
  EXPECT_EQ(expectedMemsetOutput, *dst);

  EXPECT_CUDA_SUCCESS(cudaFreeHost(src));
  EXPECT_CUDA_SUCCESS(cudaFreeHost(dst));
  EXPECT_CUDA_SUCCESS(cudaFree(device));
}

TEST(CudartDeviceMemoryManagement, Memcpy2D) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 3;
  constexpr int kSpitch = 8;
  constexpr int kDpitch = 6;
  constexpr int kSrcLength = kHeight * kSpitch;
  constexpr int kDstLength = kHeight * kDpitch;
  constexpr int kSrcSize = kSrcLength * sizeof(int);
  constexpr int kDstSize = kDstLength * sizeof(int);

  int src[kSrcSize];
  int dst[kDstSize];
  for (int i = 0; i < kSrcLength; ++i) {
    src[i] = i;
  }
  for (int i = 0; i < kDstLength; ++i) {
    dst[i] = 42;
  }
  int *device;
  EXPECT_CUDA_SUCCESS(cudaMalloc(&device, kDstSize));
  EXPECT_CUDA_SUCCESS(cudaMemcpy2D(device, kDpitch * sizeof(int), src,
                                   kSpitch * sizeof(int), kWidth * sizeof(int),
                                   kHeight, cudaMemcpyHostToDevice));
  EXPECT_CUDA_SUCCESS(
      cudaMemcpy(dst, device, kDstSize, cudaMemcpyDeviceToHost));
  for (int i = 0; i < kHeight; ++i) {
    for (int j = 0; j < kWidth; ++j) {
      int srcIndex = j + (kSpitch * i);
      int dstIndex = j + (kDpitch * i);
      EXPECT_EQ(dst[dstIndex], src[srcIndex]) << "row " << i << ", column "
                                              << j;
    }
    for (int j = kWidth; j < kDpitch; ++j) {
      int dstIndex = j + (kDpitch * i);
      EXPECT_EQ(0, dst[dstIndex]) << "row " << i << ", column " << j;
    }
  }
}

}  // namespace
