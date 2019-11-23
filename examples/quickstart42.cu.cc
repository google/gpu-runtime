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

#include <cassert>
#include <iostream>

#define CHECK_EQ(a, b) assert((a) == (b))

__global__ void Return42(int *out) { out[threadIdx.x] = 42; }

int main(int argc, char *argv[]) {
  constexpr int kDataLen = 256;
  constexpr int kArrayByteSize = kDataLen * sizeof(int);

  int hostArray[kDataLen];

  int *deviceArray;
  CHECK_EQ(cudaMalloc(&deviceArray, kArrayByteSize), cudaSuccess);

  Return42<<<1, kDataLen>>>(deviceArray);

  CHECK_EQ(cudaMemcpy(hostArray, deviceArray, kArrayByteSize,
                      cudaMemcpyDeviceToHost),
           cudaSuccess);
  for (int i = 0; i < kDataLen; ++i) {
    CHECK_EQ(42, hostArray[i]);
  }

  CHECK_EQ(cudaFree(deviceArray), cudaSuccess);

  std::cout << "All good!\n";
  return 0;
}
