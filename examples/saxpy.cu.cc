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

#include <memory>
#include <random>
#include <iostream>
#include <cassert>

#define CHECK_EQ(a, b) assert((a) == (b))
#define CHECK(a) assert(a)

// SAXPY kernel.
__global__ void CudaSaxpy(int n, float a, float *x, float *y) {
  int arrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (arrayIndex < n) {
    y[arrayIndex] += a * x[arrayIndex];
  }
}

// Generic AXPY kernel for any numerical data type.
template <typename T>
__global__ void GenericAxpy(int n, T a, T *x, T *y) {
  int arrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (arrayIndex < n) {
    y[arrayIndex] += a * x[arrayIndex];
  }
}

// Creates an array of floats randomly set to values in the range [0.0, 1.0).
std::unique_ptr<float[]> CreateRandomDataArray(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0f, 1.0f);
  std::unique_ptr<float[]> data(new float[size]);
  for (int i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }
  return data;
}

// Returns true if and only if the result data matches the SAXPY results
// computed on the host for parameters a, x, and y.
bool VerifySaxpy(int n, float a, const float *x, const float *y,
                 const float *result) {
  for (int i = 0; i < n; ++i) {
    float expected = a * x[i] + y[i];
    if (std::abs(result[i] - expected) > 0.001f) {
      std::cout << "Mismatch at i = " << i << ": " << result[i]
                << " != " << expected;
      return false;
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  constexpr int kDataLen = 1 << 25;
  constexpr int kDataByteSize = kDataLen * sizeof(float);

  std::cout.setf(std::ios::unitbuf);

  // Create host data for the SAXPY computation.
  float a = 7.0f;
  std::unique_ptr<float[]> hostX = CreateRandomDataArray(kDataLen);
  std::unique_ptr<float[]> hostY = CreateRandomDataArray(kDataLen);

  // Allocate space on the device for x and y.
  float *devX;
  float *devY;
  CHECK_EQ(cudaMalloc(&devX, kDataByteSize), cudaSuccess);
  CHECK_EQ(cudaMalloc(&devY, kDataByteSize), cudaSuccess);

  // Copy x and y to the device.
  CHECK_EQ(cudaMemcpy(devX, hostX.get(), kDataByteSize, cudaMemcpyHostToDevice),
           cudaSuccess);
  CHECK_EQ(cudaMemcpy(devY, hostY.get(), kDataByteSize, cudaMemcpyHostToDevice),
           cudaSuccess);

  // Launch the float-specific SAXPY kernel.
  int threadCount = 128;
  int blockCount = (kDataLen + threadCount - 1) / threadCount;
  std::cout << "Executing SAXPY kernel...\n";
  CudaSaxpy<<<blockCount, threadCount>>>(kDataLen, a, devX, devY);

  // Bring the results back to the host.
  auto hostResult = std::unique_ptr<float[]>(new float[kDataLen]);
  CHECK_EQ(
      cudaMemcpy(hostResult.get(), devY, kDataByteSize, cudaMemcpyDeviceToHost),
      cudaSuccess);

  // Verify the results on the host.
  std::cout << "Verifying SAXPY...\n";
  CHECK(VerifySaxpy(kDataLen, a, hostX.get(), hostY.get(), hostResult.get()));

  // Repeat the same computation using the GenericAxpy kernel.

  // Rewrite the y data on the device because it was overwritten by the last
  // kernel call.
  CHECK_EQ(cudaMemcpy(devY, hostY.get(), kDataByteSize, cudaMemcpyHostToDevice),
           cudaSuccess);

  // Call the generic kernel.
  GenericAxpy<float><<<blockCount, threadCount>>>(kDataLen, a, devX, devY);

  // Bring the results back to the host.
  CHECK_EQ(
      cudaMemcpy(hostResult.get(), devY, kDataByteSize, cudaMemcpyDeviceToHost),
      cudaSuccess);

  // Verify the results on the host.
  std::cout << "Verifying generic AXPY (float)...\n";
  CHECK(VerifySaxpy(kDataLen, a, hostX.get(), hostY.get(), hostResult.get()));

  // Free device memory.
  CHECK_EQ(cudaFree(devX), cudaSuccess);
  CHECK_EQ(cudaFree(devY), cudaSuccess);

  std::cout << "Done.\n";

  return 0;
}
