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

#include <string>
#include <cuda_runtime.h>
#include "gtest/gtest.h"
#include "tests/common.h"

// Kernels and device-side globals must be in the same TU in order to access
// host-side 'shadow' copies.
#include "tests/device_code.cu"

// These tests currently do not work because host-side shadow variables are not
// externally visible and can't be accessed from a different TU. Original
// instance of this test used to load .fatbin manually and thus were not
// affected by this.

namespace {

static const char *symbolName = "deviceArray";
static const char* nestedSymbolName =
    "::gpu_runtime::testing::nestedDeviceArray";

TEST(CudartGlobalSymbolManagementTest, BasicUsage) {
  void *devPtr;
  EXPECT_CUDA_SUCCESS(cudaGetSymbolAddress(&devPtr, deviceArray));

  size_t size;
  EXPECT_CUDA_SUCCESS(cudaGetSymbolSize(&size, deviceArray));
  EXPECT_EQ(size, 4 * sizeof(int));
}

TEST(CudartGlobalSymbolManagementTest, SymbolInNamespace) {
  void *devPtr;
  EXPECT_CUDA_SUCCESS(
      cudaGetSymbolAddress(&devPtr, gpu_runtime::testing::nestedDeviceArray));

  size_t size;
  EXPECT_CUDA_SUCCESS(
      cudaGetSymbolSize(&size, gpu_runtime::testing::nestedDeviceArray));
  EXPECT_EQ(size, 4 * sizeof(int));
}

TEST(CudartGlobalSymbolManagementTest, Memcpy) {
  int start[] = {1, 2, 3, 4};
  int finish[] = {0, 0, 0, 0};
  EXPECT_CUDA_SUCCESS(
      cudaMemcpyToSymbol(deviceArray, start, 4 * sizeof(int), 0));
  EXPECT_CUDA_SUCCESS(
      cudaMemcpyFromSymbol(finish, deviceArray, 4 * sizeof(int), 0));
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(i + 1, finish[i]) << "mismatch at position " << i;
  }
}

TEST(CudartGlobalSymbolManagementTest, MemcpyAsync) {
  int start[] = {1, 2, 3, 4};
  int finish[] = {0, 0, 0, 0};
  cudaStream_t stream;
  EXPECT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  EXPECT_CUDA_SUCCESS(cudaMemcpyToSymbolAsync(
      deviceArray, start, 4 * sizeof(int), 0, cudaMemcpyHostToDevice, stream));
  EXPECT_CUDA_SUCCESS(cudaMemcpyFromSymbolAsync(
      finish, deviceArray, 4 * sizeof(int), 0, cudaMemcpyDeviceToHost, stream));
  EXPECT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(i + 1, finish[i]) << "mismatch at position " << i;
  }
}

TEST(CudartGlobalSymbolManagementTest, MemcpyOffset) {
  int start = 42;
  int finish[] = {0, 0, 0, 0};
  EXPECT_CUDA_SUCCESS(
      cudaMemcpyToSymbol(deviceArray, &start, sizeof(int), 2 * sizeof(int)));
  EXPECT_CUDA_SUCCESS(
      cudaMemcpyFromSymbol(finish, deviceArray, 4 * sizeof(int), 0));
  EXPECT_EQ(42, finish[2]);
}

TEST(CudartGlobalSymbolManagementTest, MemcpyAsyncOffset) {
  int start[] = {1, 2, 3, 4};
  int finish = 0;
  cudaStream_t stream;
  EXPECT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  EXPECT_CUDA_SUCCESS(cudaMemcpyToSymbolAsync(
      deviceArray, start, 4 * sizeof(int), 0, cudaMemcpyHostToDevice, stream));
  EXPECT_CUDA_SUCCESS(cudaMemcpyFromSymbolAsync(
      &finish, deviceArray, sizeof(int), 2 * sizeof(int),
      cudaMemcpyDeviceToHost, stream));
  EXPECT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  EXPECT_EQ(3, finish);
}

}  // namespace
