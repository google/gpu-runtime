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

#ifndef PERFTOOLS_GPUTOOLS_GCUDART_TESTS_COMMON_H_
#define PERFTOOLS_GPUTOOLS_GCUDART_TESTS_COMMON_H_

// Macros for checking cudaSuccess
#define EXPECT_CUDA_SUCCESS(expr) EXPECT_EQ(cudaSuccess, expr)
#define ASSERT_CUDA_SUCCESS(expr) ASSERT_EQ(cudaSuccess, expr)

// Struct used to register a fat binary with cudart.
struct FatCubin {
  int magic_number = 0x466243b1;
  int version = 1;
  const void *binary;  // fatbin data
  void *unused;
};

#endif  // PERFTOOLS_GPUTOOLS_GCUDART_TESTS_COMMON_H_
