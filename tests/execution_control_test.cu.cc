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

#include <vector>

#include <cuda_runtime.h>
#include "tests/common.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "absl/strings/str_cat.h"

#include "device_code.cu"

namespace {

TEST(CudartExecutionControlTest, FunctionAttributes) {
  cudaFuncAttributes attributes;

  // Set some dummy values to be overridden by querying the real attributes.
  attributes.constSizeBytes = -1;
  attributes.localSizeBytes = -1;
  attributes.ptxVersion = -1;
  attributes.sharedSizeBytes = -1;
  attributes.numRegs = -1;

  EXPECT_CUDA_SUCCESS(
      cudaFuncGetAttributes(&attributes, (const void*)addKernel));
  EXPECT_GE(attributes.constSizeBytes, 0);
  EXPECT_GE(attributes.localSizeBytes, 0);
  EXPECT_GE(attributes.sharedSizeBytes, 0);

  EXPECT_THAT(attributes.ptxVersion, testing::AnyOf(30, 32, 35, 37, 50, 52, 53,
                                                    60, 61, 62, 70, 72, 75));
  EXPECT_GT(attributes.maxThreadsPerBlock, 0);
  EXPECT_LE(attributes.maxThreadsPerBlock, 1024);

  EXPECT_GT(attributes.numRegs, 0);
  EXPECT_LE(attributes.numRegs, 255);
}

TEST(CudartExecutionControlTest, LaunchKernel) {
  int hostA = 1;
  int hostB = 2;

  int *deviceA;
  int *deviceB;

  EXPECT_CUDA_SUCCESS(cudaMalloc(&deviceA, sizeof(int)));
  EXPECT_CUDA_SUCCESS(cudaMalloc(&deviceB, sizeof(int)));
  EXPECT_CUDA_SUCCESS(
      cudaMemcpy(deviceA, &hostA, sizeof(int), cudaMemcpyHostToDevice));
  EXPECT_CUDA_SUCCESS(
      cudaMemcpy(deviceB, &hostB, sizeof(int), cudaMemcpyHostToDevice));

  std::vector<void *> args;
  args.push_back(&deviceA);
  args.push_back(&deviceB);
  dim3 gridDim(1, 1, 1);
  dim3 blockDim(1, 1, 1);
  EXPECT_CUDA_SUCCESS(
      cudaLaunchKernel((const void*)addKernel, gridDim, blockDim, args.data()));

  EXPECT_CUDA_SUCCESS(
      cudaMemcpy(&hostA, deviceA, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_CUDA_SUCCESS(
      cudaMemcpy(&hostB, deviceB, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(3, hostA);
  EXPECT_EQ(2, hostB);
}

TEST(CudartExecutionControlTest, MultiDimensionalLaunch) {
  dim3 blockDim(5, 3, 2);
  dim3 gridDim(4, 6, 3);

  int count =
      blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
  size_t intSize = sizeof(int);

  std::unique_ptr<int[]> hostThreadIdsX(new int[count]);
  std::unique_ptr<int[]> hostThreadIdsY(new int[count]);
  std::unique_ptr<int[]> hostThreadIdsZ(new int[count]);
  std::unique_ptr<int[]> hostBlockIdsX(new int[count]);
  std::unique_ptr<int[]> hostBlockIdsY(new int[count]);
  std::unique_ptr<int[]> hostBlockIdsZ(new int[count]);

  int *deviceThreadIdsX;
  int *deviceThreadIdsY;
  int *deviceThreadIdsZ;
  int *deviceBlockIdsX;
  int *deviceBlockIdsY;
  int *deviceBlockIdsZ;

  EXPECT_CUDA_SUCCESS(cudaMalloc(&deviceThreadIdsX, count * intSize));
  EXPECT_CUDA_SUCCESS(cudaMalloc(&deviceThreadIdsY, count * intSize));
  EXPECT_CUDA_SUCCESS(cudaMalloc(&deviceThreadIdsZ, count * intSize));
  EXPECT_CUDA_SUCCESS(cudaMalloc(&deviceBlockIdsX, count * intSize));
  EXPECT_CUDA_SUCCESS(cudaMalloc(&deviceBlockIdsY, count * intSize));
  EXPECT_CUDA_SUCCESS(cudaMalloc(&deviceBlockIdsZ, count * intSize));

  std::array<void *, 6> args = {{&deviceThreadIdsX, &deviceThreadIdsY,
                                 &deviceThreadIdsZ, &deviceBlockIdsX,
                                 &deviceBlockIdsY, &deviceBlockIdsZ}};
  EXPECT_CUDA_SUCCESS(cudaLaunchKernel((const void*)reportThreadsKernel,
                                       gridDim, blockDim, args.data()));

  EXPECT_CUDA_SUCCESS(cudaMemcpy(hostThreadIdsX.get(), deviceThreadIdsX,
                                 count * intSize, cudaMemcpyDeviceToHost));
  EXPECT_CUDA_SUCCESS(cudaMemcpy(hostThreadIdsY.get(), deviceThreadIdsY,
                                 count * intSize, cudaMemcpyDeviceToHost));
  EXPECT_CUDA_SUCCESS(cudaMemcpy(hostThreadIdsZ.get(), deviceThreadIdsZ,
                                 count * intSize, cudaMemcpyDeviceToHost));
  EXPECT_CUDA_SUCCESS(cudaMemcpy(hostBlockIdsX.get(), deviceBlockIdsX,
                                 count * intSize, cudaMemcpyDeviceToHost));
  EXPECT_CUDA_SUCCESS(cudaMemcpy(hostBlockIdsY.get(), deviceBlockIdsY,
                                 count * intSize, cudaMemcpyDeviceToHost));
  EXPECT_CUDA_SUCCESS(cudaMemcpy(hostBlockIdsZ.get(), deviceBlockIdsZ,
                                 count * intSize, cudaMemcpyDeviceToHost));

  for (int i = 0; i < count; ++i) {
    int expectedThreadIdX = i % blockDim.x;
    int expectedThreadIdY = (i / blockDim.x) % blockDim.y;
    int expectedThreadIdZ = (i / (blockDim.x * blockDim.y)) % blockDim.z;
    int expectedBlockIdX =
        (i / (blockDim.x * blockDim.y * blockDim.z)) % gridDim.x;
    int expectedBlockIdY =
        (i / (blockDim.x * blockDim.y * blockDim.z * gridDim.x)) % gridDim.y;
    int expectedBlockIdZ =
        i / (blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y) %
        gridDim.z;

    SCOPED_TRACE(absl::StrCat("position ", i));

    EXPECT_EQ(expectedThreadIdX, hostThreadIdsX[i]);
    EXPECT_EQ(expectedThreadIdY, hostThreadIdsY[i]);
    EXPECT_EQ(expectedThreadIdZ, hostThreadIdsZ[i]);
    EXPECT_EQ(expectedBlockIdX, hostBlockIdsX[i]);
    EXPECT_EQ(expectedBlockIdY, hostBlockIdsY[i]);
    EXPECT_EQ(expectedBlockIdZ, hostBlockIdsZ[i]);
  }
}

}  // namespace
