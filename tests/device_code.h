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

// An example global device symbol.
extern __device__ int deviceArray[4];

namespace gpu_runtime {
namespace testing {
// An example of a global device symbol in a namespace.
extern __device__ int nestedDeviceArray[4];
}  // namespace testing
}  // namespace gpu_runtime

// Add two integer arrays and store the result in the first array.
//
// Assumes blockDim.y = blockDim.z = gridDim.x = gridDim.y = gridDim.z = 1.
extern __global__ void addKernel(int* a, int* b);

// Reports which threads and blocks ran this kernel.
extern __global__ void reportThreadsKernel(int* threadIdsX, int* threadIdsY,
                                    int* threadIdsZ, int* blockIdsX,
                                           int* blockIdsY, int* blockIdsZ);
