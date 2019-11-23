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
#include "absl/synchronization/mutex.h"

namespace {

TEST(CudartStreamManagement, BasicUsage) {
  // Create a stream.
  cudaStream_t stream;
  EXPECT_CUDA_SUCCESS(cudaStreamCreate(&stream));

  // Check the default stream flags.
  unsigned int streamFlags = 42u;
  EXPECT_CUDA_SUCCESS(cudaStreamGetFlags(stream, &streamFlags));
  EXPECT_EQ(0u, streamFlags);

  // Check the default priority.
  int streamPriority = 42;
  EXPECT_CUDA_SUCCESS(cudaStreamGetPriority(stream, &streamPriority));
  EXPECT_EQ(0, streamPriority);

  // Check that we can query the stream.
  EXPECT_CUDA_SUCCESS(cudaStreamQuery(stream));

  // Check that we can destroy the stream.
  EXPECT_CUDA_SUCCESS(cudaStreamDestroy(stream));

  // Create a stream with the non-blocking flags.
  cudaStream_t nonblockingStream;
  EXPECT_CUDA_SUCCESS(
      cudaStreamCreateWithFlags(&nonblockingStream, cudaStreamNonBlocking));
  unsigned int nonblockingStreamFlags = 42u;
  EXPECT_CUDA_SUCCESS(
      cudaStreamGetFlags(nonblockingStream, &nonblockingStreamFlags));
  EXPECT_EQ(cudaStreamNonBlocking, static_cast<int>(nonblockingStreamFlags));
  EXPECT_CUDA_SUCCESS(cudaStreamDestroy(nonblockingStream));

  // Check that we can get the allowable stream priorities.
  int leastPriority;
  int greatestPriority;
  EXPECT_CUDA_SUCCESS(
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

  // Check that we can create a stream with a priority.
  cudaStream_t priorityStream;
  int requestedPriority = leastPriority;
  int queriedPriority = 42;
  EXPECT_CUDA_SUCCESS(cudaStreamCreateWithPriority(
      &priorityStream, cudaStreamDefault, requestedPriority));
  EXPECT_CUDA_SUCCESS(cudaStreamGetPriority(priorityStream, &queriedPriority));
  EXPECT_CUDA_SUCCESS(cudaStreamDestroy(priorityStream));
}

// This callback interprets the userData as a pointer to an integer and sets the
// value of that integer to 42.
//
// \param stream The stream in which the callback is running.
// \param status The status of the stream in which the callback is running.
// \param userData Arbitrary user data to pass to the callback.
void CUDART_CB setValueTo42(cudaStream_t, cudaError_t, void *userData) {
  int *intPointer = reinterpret_cast<int *>(userData);
  *intPointer = 42;
}

TEST(CudartStreamManagement, BasicCallback) {
  cudaStream_t stream;
  EXPECT_CUDA_SUCCESS(cudaStreamCreate(&stream));

  // This is the integer that the callback will set to 42.
  int callbackInteger = 0;

  // Enqueue the callback in the stream.
  EXPECT_CUDA_SUCCESS(
      cudaStreamAddCallback(stream, setValueTo42, &callbackInteger, 0u));

  // Synchronize on the stream to make sure all its work is complete.
  EXPECT_CUDA_SUCCESS(cudaStreamSynchronize(stream));

  // Check that the callback really ran and set the value to 42.
  EXPECT_EQ(42, callbackInteger);
  EXPECT_CUDA_SUCCESS(cudaStreamDestroy(stream));
}

// User data for StreamWaitEvent test below.
struct StreamWaitEventUserData {
  absl::Mutex mutex;
  bool firstStreamCanStart ABSL_GUARDED_BY(mutex) = false;
  bool firstStreamDoneFlag ABSL_GUARDED_BY(mutex) = false;
};

// Waits for userData->firstStreamCanStart and sets
// userData->firstStreamDoneFlag.
//
// Used in the StreamWaitEvent test below.
void CUDART_CB SetFirstStreamDoneFlag(cudaStream_t stream, cudaError_t status,
                                      void *userData) {
  auto *data = static_cast<StreamWaitEventUserData *>(userData);
  absl::MutexLock Lock(&data->mutex);
  EXPECT_FALSE(data->firstStreamDoneFlag);
  data->mutex.Await(absl::Condition(&data->firstStreamCanStart));
  data->firstStreamDoneFlag = true;
}

// Checks that userData->firstStreamDoneFlag is true.
//
// Used in the StreamWaitEvent test below.
void CUDART_CB CheckFirstStreamDoneFlag(cudaStream_t stream, cudaError_t status,
                                        void *userData) {
  auto *data = static_cast<StreamWaitEventUserData *>(userData);
  absl::MutexLock Lock(&data->mutex);
  EXPECT_TRUE(data->firstStreamDoneFlag);
}

TEST(CudartStreamManagement, StreamWaitEvent) {
  cudaStream_t firstStream;
  cudaStream_t secondStream;
  cudaEvent_t firstStreamDoneEvent;

  EXPECT_CUDA_SUCCESS(cudaStreamCreate(&firstStream));
  EXPECT_CUDA_SUCCESS(cudaStreamCreate(&secondStream));
  EXPECT_CUDA_SUCCESS(cudaEventCreate(&firstStreamDoneEvent));

  StreamWaitEventUserData userData;

  // The firstStream waits for userData.firstStreamCanStart to be signalled,
  // then sets userData.firstStreamDoneFlag, then records the
  // firstStreamDoneEvent.
  EXPECT_CUDA_SUCCESS(
      cudaStreamAddCallback(firstStream, SetFirstStreamDoneFlag, &userData, 0));
  EXPECT_CUDA_SUCCESS(cudaEventRecord(firstStreamDoneEvent, firstStream));

  // The secondStream waits for the firstStreamDoneEvent to be recorded in
  // firstStream, then checks that userData.firstStreamDoneFlag is set. This
  // checks that the call to cudaStreamWaitEvent has really made the
  // secondStream wait to run until the firstStream is done.
  EXPECT_CUDA_SUCCESS(
      cudaStreamWaitEvent(secondStream, firstStreamDoneEvent, 0));
  EXPECT_CUDA_SUCCESS(cudaStreamAddCallback(
      secondStream, CheckFirstStreamDoneFlag, &userData, 0));

  // The secondStream could be running here if we hadn't used
  // cudaStreamWaitEvent to make it wait.

  // Wake the callback running in firstStream.
  {
    absl::MutexLock Lock(&userData.mutex);
    userData.firstStreamCanStart = true;
  }

  EXPECT_CUDA_SUCCESS(cudaStreamDestroy(firstStream));
  EXPECT_CUDA_SUCCESS(cudaStreamDestroy(secondStream));
  EXPECT_CUDA_SUCCESS(cudaEventDestroy(firstStreamDoneEvent));
}

}  // namespace
