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

#include "tests/common.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

__global__ void SwapPairs(int *inOut, int length) {
  extern __shared__ int scratch[];
  if (threadIdx.x < (length & ~1)) {
    scratch[threadIdx.x] = inOut[threadIdx.x];
    __syncthreads();
    inOut[threadIdx.x] = scratch[threadIdx.x ^ 1];
  }
}

namespace {

TEST(DynamicSharedMemory, SwapPairs) {
  constexpr int kLength = 5;
  constexpr int kSize = sizeof(int) * kLength;

  int host[kLength] = {0, 1, 2, 3, 4};
  int *device;
  EXPECT_CUDA_SUCCESS(cudaMalloc(&device, kSize));
  EXPECT_CUDA_SUCCESS(cudaMemcpy(device, host, kSize, cudaMemcpyHostToDevice));

  SwapPairs<<<1, kLength, kSize>>>(device, kLength);

  EXPECT_CUDA_SUCCESS(cudaMemcpy(host, device, kSize, cudaMemcpyDeviceToHost));
  EXPECT_THAT(host, testing::ElementsAre(1, 0, 3, 2, 4));
}

}  // namespace

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
