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
class CudartDeviceManagement : public ::testing::TestWithParam<int> {};

TEST(CudartDeviceManagement, BasicUsage) {
  // Check that default device has ordinal 0.
  int deviceOrdinal;
  EXPECT_CUDA_SUCCESS(cudaGetDevice(&deviceOrdinal));
  EXPECT_EQ(0, deviceOrdinal);
  EXPECT_CUDA_SUCCESS(cudaPeekAtLastError());
  EXPECT_CUDA_SUCCESS(cudaGetLastError());

  // Check that we can get the device count.
  int deviceCount;
  EXPECT_CUDA_SUCCESS(cudaGetDeviceCount(&deviceCount));
  EXPECT_CUDA_SUCCESS(cudaPeekAtLastError());
  EXPECT_CUDA_SUCCESS(cudaGetLastError());

  // Check that we can set and get each device ordinal, synchronize on each
  // device, and repeatedly reset each device.
  for (int i = deviceCount - 1; i >= 0; --i) {
    EXPECT_CUDA_SUCCESS(cudaSetDevice(i));
    EXPECT_CUDA_SUCCESS(cudaGetDevice(&deviceOrdinal));
    EXPECT_EQ(i, deviceOrdinal);
    EXPECT_CUDA_SUCCESS(cudaPeekAtLastError());
    EXPECT_CUDA_SUCCESS(cudaGetLastError());
    EXPECT_CUDA_SUCCESS(cudaDeviceSynchronize());
    EXPECT_CUDA_SUCCESS(cudaDeviceReset());
    EXPECT_CUDA_SUCCESS(cudaDeviceReset());
  }

  // Try to set a device ordinal that is too high.
  EXPECT_EQ(cudaErrorInvalidDevice, cudaSetDevice(deviceCount));

  // Make sure the last error peek and get functions work.
  EXPECT_EQ(cudaErrorInvalidDevice, cudaPeekAtLastError());
  EXPECT_EQ(cudaErrorInvalidDevice, cudaGetLastError());

  // Make sure that getting the last error clears the last error.
  EXPECT_CUDA_SUCCESS(cudaPeekAtLastError());
  EXPECT_CUDA_SUCCESS(cudaGetLastError());
}

TEST(CudartDeviceManagement, GetAttribute) {
  int value;
  for (int i = 1; i < 86; ++i) {
    EXPECT_CUDA_SUCCESS(
        cudaDeviceGetAttribute(&value, static_cast<cudaDeviceAttr>(i), 0))
        << "attribute number " << i;
  }
}

#if 0

// FIXME: These are currently failing with CUDA_ERROR_CONTEXT_IS_DESTROYED. No
// idea why.

TEST_P(CudartDeviceManagement, SetOneFlag) {
  unsigned int flag = GetParam();
  EXPECT_CUDA_SUCCESS(cudaDeviceReset());
  EXPECT_CUDA_SUCCESS(cudaSetDeviceFlags(flag));

  // Perform some operation that forces the context to be set for the device.
  // We do cudaMalloc here, but it could have been many other operations such
  // as cudaMemcpy or a kernel launch.
  char* dummyBuffer;
  EXPECT_CUDA_SUCCESS(cudaMalloc(&dummyBuffer, 1));

  // Check that the flags were set to the right values.
  unsigned int measured = 0;
  EXPECT_CUDA_SUCCESS(cudaGetDeviceFlags(&measured));
  EXPECT_EQ(measured, flag | cudaDeviceMapHost);

  // Check that trying to set the flags again fails.
  EXPECT_EQ(cudaSetDeviceFlags(flag), cudaErrorSetOnActiveProcess);
  EXPECT_EQ(cudaGetLastError(), cudaErrorSetOnActiveProcess);

  // Reset and set the flags again.
  EXPECT_CUDA_SUCCESS(cudaDeviceReset());
  EXPECT_CUDA_SUCCESS(cudaGetDeviceFlags(&measured));
  EXPECT_EQ(measured, 0);
  EXPECT_CUDA_SUCCESS(cudaSetDeviceFlags(flag));
  EXPECT_CUDA_SUCCESS(cudaMalloc(&dummyBuffer, 1));
  EXPECT_CUDA_SUCCESS(cudaGetDeviceFlags(&measured));
  EXPECT_EQ(measured, flag | cudaDeviceMapHost);

}
INSTANTIATE_TEST_SUITE_P(SetDeviceFlags, CudartDeviceManagement,
                         testing::Values(cudaDeviceScheduleAuto,
                                         cudaDeviceScheduleSpin,
                                         cudaDeviceScheduleYield,
                                         cudaDeviceScheduleBlockingSync));
#endif
}  // namespace
