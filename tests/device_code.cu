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
__device__ int deviceArray[4];
int hostArray[4];

namespace gpu_runtime {
namespace testing {
// An example of a global device symbol in a namespace.
__device__ int nestedDeviceArray[4];
}  // namespace testing
}  // namespace gpu_runtime

// Add two integer arrays and store the result in the first array.
//
// Assumes blockDim.y = blockDim.z = gridDim.x = gridDim.y = gridDim.z = 1.
__global__ void addKernel(int* a, int* b) { a[threadIdx.x] += b[threadIdx.x]; }

// Reports which threads and blocks ran this kernel.
__global__ void reportThreadsKernel(int* threadIdsX, int* threadIdsY,
                                    int* threadIdsZ, int* blockIdsX,
                                    int* blockIdsY, int* blockIdsZ) {
  int threadXStride = 1;
  int threadYStride = threadXStride * blockDim.x;
  int threadZStride = threadYStride * blockDim.y;
  int blockXStride = threadZStride * blockDim.z;
  int blockYStride = blockXStride * gridDim.x;
  int blockZStride = blockYStride * gridDim.y;

  int tid = (threadXStride * threadIdx.x) + (threadYStride * threadIdx.y) +
            (threadZStride * threadIdx.z) + (blockXStride * blockIdx.x) +
            (blockYStride * blockIdx.y) + (blockZStride * blockIdx.z);

  threadIdsX[tid] = threadIdx.x;
  threadIdsY[tid] = threadIdx.y;
  threadIdsZ[tid] = threadIdx.z;
  blockIdsX[tid] = blockIdx.x;
  blockIdsY[tid] = blockIdx.y;
  blockIdsZ[tid] = blockIdx.z;
}
