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
#include "absl/strings/str_cat.h"

// Check measured result for expected error and clear the last error.
#define EXPECT_CUDA_ERROR(expected, measured)      \
  do {                                             \
    cudaError_t expectedResult = expected;         \
    EXPECT_EQ(measured, expectedResult);           \
    EXPECT_EQ(cudaGetLastError(), expectedResult); \
  } while (false)

namespace {

TEST(PeerDeviceMemoryAccessTest, InvalidDevices) {
  int deviceCount = 0;
  EXPECT_CUDA_SUCCESS(cudaGetDeviceCount(&deviceCount));
  EXPECT_GT(deviceCount, 0);

  for (int i = 0; i < deviceCount; ++i) {
    SCOPED_TRACE(absl::StrCat("primary device = ", i));
    int canAccessPeer = 0;

    // Can access peer
    EXPECT_CUDA_ERROR(cudaErrorInvalidDevice,
                      cudaDeviceCanAccessPeer(&canAccessPeer, i, -1));
    EXPECT_CUDA_ERROR(cudaErrorInvalidDevice,
                      cudaDeviceCanAccessPeer(&canAccessPeer, i, deviceCount));

    EXPECT_CUDA_SUCCESS(cudaSetDevice(i));

    // Enable peer access
    EXPECT_CUDA_ERROR(cudaErrorInvalidDevice,
                      cudaDeviceEnablePeerAccess(-1, 0));
    EXPECT_CUDA_ERROR(cudaErrorInvalidDevice,
                      cudaDeviceEnablePeerAccess(deviceCount, 0));
    EXPECT_CUDA_ERROR(cudaErrorInvalidValue, cudaDeviceEnablePeerAccess(0, 1));

    // Disable peer access
    EXPECT_CUDA_ERROR(cudaErrorInvalidDevice, cudaDeviceDisablePeerAccess(-1));
    EXPECT_CUDA_ERROR(cudaErrorInvalidDevice,
                      cudaDeviceDisablePeerAccess(deviceCount));
  }
}

}  // namespace
