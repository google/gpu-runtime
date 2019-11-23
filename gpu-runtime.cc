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

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

// Treat cuResultExpr as a CUresult and return it if it is not CUDA_SUCCESS.
#define RETURN_IF_FAIL(cuResultExpr) \
  do {                               \
    CUresult error = (cuResultExpr); \
    if (error) {                     \
      return error;                  \
    }                                \
  } while (false)

// Treat cudaErrorExpr as a cudaError_t and return it if it is an error.
// Otherwise, return the last error.
#define RETURN_CUDART(cudaErrorExpr)            \
  do {                                          \
    cudaError_t error = (cudaErrorExpr);        \
    if (error) {                                \
      threadState.lastError = error;            \
    }                                           \
    return threadState.lastError;               \
  } while (false)

// Treat cudaErrorExpr as a cudaError_t and return it if it is not cudaSuccess.
#define RETURN_CUDART_IF_ERROR(cudaErrorExpr) \
  do {                                        \
    cudaError_t error = (cudaErrorExpr);      \
    if (error) {                              \
      threadState.lastError = error;          \
      return threadState.lastError;           \
    }                                         \
  } while (false)

// Treat cuResultExpr as a CUresult and if it is not CUDA_SUCCESS, convert it
// to a cudaError_t and return that.
//
// Also sets threadState.lastError if not CUDA_SUCCESS.
#define RETURN_CONVERTED_IF_FAIL(cuResultExpr)              \
  do {                                                      \
    cudaError_t error = convertToCudartError(cuResultExpr); \
    if (error) {                                            \
      threadState.lastError = error;                        \
      return threadState.lastError;                         \
    }                                                       \
  } while (false)

// Treat cuResultExpr as a CUresult, convert it to a cudaError_t and return
// that cudaError_t value.
//
// Also sets threadState.lastError if not CUDA_SUCCESS.
#define RETURN_CONVERTED(cuResultExpr)                      \
  do {                                                      \
    cudaError_t error = convertToCudartError(cuResultExpr); \
    if (error) {                                            \
      threadState.lastError = error;                        \
    }                                                       \
    return threadState.lastError;                           \
  } while (false)

namespace {

using MutexLock = std::lock_guard<std::mutex>;

typedef CUstream_st *CUstream;
typedef CUevent_st *CUevent;

static CUresult RetainPrimaryContext(CUcontext *context, CUdevice device) {
  return cuDevicePrimaryCtxRetain(context, device);
}

// State maintained per thread.
struct ThreadState {
  int activeDeviceOrdinal = 0;
  bool isInitialized = false;
  cudaError_t lastError = cudaSuccess;
  CUcontext activeContext = nullptr;
};


//STATIC_THREAD_LOCAL(ThreadState, threadState);
thread_local ThreadState threadState;

// State returned from initializing this library.
class InitializationState {
 public:
  InitializationState() {
    resultCode_ = cuInit(0);
    if (resultCode_) {
      return;
    }

    deviceCount_ = 0;
    resultCode_ = cuDeviceGetCount(&deviceCount_);
    if (resultCode_) {
      return;
    }
  }

  std::pair<CUcontext, CUresult> GetContext(int deviceOrdinal) {
    if (resultCode_) {
      return {nullptr, resultCode_};
    }
    if (!IsValidDevice(deviceOrdinal)) {
      return {nullptr, CUDA_ERROR_INVALID_DEVICE};
    }
    MutexLock Lock(mutex_);
    auto iterator = primaryContextMap_.find(deviceOrdinal);
    if (iterator != primaryContextMap_.end()) {
      return {iterator->second, CUDA_SUCCESS};
    }
    CUdevice device;
    if (CUresult error = cuDeviceGet(&device, deviceOrdinal)) {
      return {nullptr, error};
    }
    CUcontext context;
    if (CUresult error = RetainPrimaryContext(&context, device)) {
      return {nullptr, error};
    }
    primaryContextMap_.emplace(device, context);
    return {context, CUDA_SUCCESS};
  }

  CUresult ResetContext(int deviceOrdinal) {
    MutexLock Lock(mutex_);
    CUdevice device;
    if (CUresult result = cuDeviceGet(&device, deviceOrdinal)) {
      return result;
    }
    primaryContextMap_.erase(device);
    return cuDevicePrimaryCtxReset(device);
  }

  // Returns true if deviceOrdinal is in the range [0, deviceCount).
  bool IsValidDevice(int deviceOrdinal) const {
    return 0 <= deviceOrdinal && deviceOrdinal < deviceCount_;
  }

  CUresult getResultCode() const { return resultCode_; }

 private:
  CUresult resultCode_;
  int deviceCount_;

  std::mutex mutex_;
  std::map<CUdevice, CUcontext> primaryContextMap_ ;//ABSL_GUARDED_BY(mutex_);
};

// State returned from initializing a thread.
//
// Has a CUresult result code and a pointer to the PrimaryContextMap.
struct ThreadInitializationState {
  CUresult resultCode;
  InitializationState *initializationState;
};

}  // namespace

// Gets a string describing a CUresult error.
static std::string GetErrorDescription(CUresult error) {
  const char *errorName;
  const char *errorString;
  if (cuGetErrorName(error, &errorName)) {
    errorName = "UNKNOWN NAME";
  }
  if (cuGetErrorString(error, &errorString)) {
    errorString = "UNKNOWN DESCRIPTION";
  }
  return "CUDA driver error code " + std::to_string(error) + ": error name = " +
      errorName + ", error description = " + errorString;
}

// Converts a CUresult to its corresponding cudaError_t.
static cudaError_t convertToCudartError(CUresult error) {
  switch (error) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE:
      return cudaErrorInvalidValue;
    case CUDA_ERROR_OUT_OF_MEMORY:
      return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED:
      return cudaErrorInitializationError;
    case CUDA_ERROR_DEINITIALIZED:
      return cudaErrorCudartUnloading;
    case CUDA_ERROR_PROFILER_DISABLED:
      return cudaErrorProfilerDisabled;
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
      return cudaErrorProfilerNotInitialized;
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
      return cudaErrorProfilerAlreadyStarted;
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
      return cudaErrorProfilerAlreadyStopped;
    case CUDA_ERROR_NO_DEVICE:
      return cudaErrorNoDevice;
    case CUDA_ERROR_INVALID_DEVICE:
      return cudaErrorInvalidDevice;
    case CUDA_ERROR_INVALID_IMAGE:
      return cudaErrorInvalidKernelImage;
    case CUDA_ERROR_INVALID_CONTEXT:
      return cudaErrorIncompatibleDriverContext;
    case CUDA_ERROR_MAP_FAILED:
      return cudaErrorMapBufferObjectFailed;
    case CUDA_ERROR_UNMAP_FAILED:
      return cudaErrorUnmapBufferObjectFailed;
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
      return cudaErrorNoKernelImageForDevice;
    case CUDA_ERROR_ECC_UNCORRECTABLE:
      return cudaErrorECCUncorrectable;
    case CUDA_ERROR_UNSUPPORTED_LIMIT:
      return cudaErrorUnsupportedLimit;
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
      return cudaErrorPeerAccessUnsupported;
    case CUDA_ERROR_INVALID_PTX:
      return cudaErrorInvalidPtx;
    case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
      return cudaErrorInvalidGraphicsContext;
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
      return cudaErrorSharedObjectSymbolNotFound;
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
      return cudaErrorSharedObjectInitFailed;
    case CUDA_ERROR_OPERATING_SYSTEM:
      return cudaErrorOperatingSystem;
    case CUDA_ERROR_INVALID_HANDLE:
      return cudaErrorInvalidResourceHandle;
    case CUDA_ERROR_NOT_FOUND:
      return cudaErrorInvalidSymbol;
    case CUDA_ERROR_NOT_READY:
      return cudaErrorNotReady;
    case CUDA_ERROR_ILLEGAL_ADDRESS:
      return cudaErrorIllegalAddress;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      return cudaErrorLaunchOutOfResources;
    case CUDA_ERROR_LAUNCH_TIMEOUT:
      return cudaErrorLaunchTimeout;
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
      return cudaErrorPeerAccessAlreadyEnabled;
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
      return cudaErrorPeerAccessNotEnabled;
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:
      return cudaErrorIncompatibleDriverContext;
    case CUDA_ERROR_ASSERT:
      return cudaErrorAssert;
    case CUDA_ERROR_TOO_MANY_PEERS:
      return cudaErrorTooManyPeers;
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
      return cudaErrorHostMemoryAlreadyRegistered;
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
      return cudaErrorHostMemoryNotRegistered;
    case CUDA_ERROR_HARDWARE_STACK_ERROR:
      return cudaErrorHardwareStackError;
    case CUDA_ERROR_ILLEGAL_INSTRUCTION:
      return cudaErrorIllegalInstruction;
    case CUDA_ERROR_MISALIGNED_ADDRESS:
      return cudaErrorMisalignedAddress;
    case CUDA_ERROR_INVALID_ADDRESS_SPACE:
      return cudaErrorInvalidAddressSpace;
    case CUDA_ERROR_INVALID_PC:
      return cudaErrorInvalidPc;
    case CUDA_ERROR_LAUNCH_FAILED:
      return cudaErrorLaunchFailure;
    case CUDA_ERROR_NOT_PERMITTED:
      return cudaErrorNotPermitted;
    case CUDA_ERROR_NOT_SUPPORTED:
      return cudaErrorNotSupported;
    case CUDA_ERROR_UNKNOWN:
      return cudaErrorUnknown;
    default:
      return cudaErrorUnknown;
  }
}

// Initializes gcudart if it has not already been initialized.
static InitializationState *initializeIfNeeded() {
  static InitializationState initializationState;
  return &initializationState;
}

// Sets the active device ordinal for this thread.
static CUresult setDeviceForThread(int deviceOrdinal,
                                   InitializationState *initializationState) {
  ThreadState *state = &threadState;
  std::pair<CUcontext, CUresult> maybeContext =
      initializationState->GetContext(deviceOrdinal);
  if (maybeContext.second) {
    return maybeContext.second;
  }
  state->activeDeviceOrdinal = deviceOrdinal;
  state->activeContext = maybeContext.first;
  return cuCtxSetCurrent(maybeContext.first);
}

// Sets up the per-thread state for this thread if not already done.
static ThreadInitializationState initializeThreadIfNeeded() {
  InitializationState *initState = initializeIfNeeded();
  ThreadInitializationState threadInitState = {initState->getResultCode(),
                                               initState};
  if (threadInitState.resultCode) {
    return threadInitState;
  }
  ThreadState *state = &threadState;
  if (!state->isInitialized) {
    if (CUresult error =
            setDeviceForThread(state->activeDeviceOrdinal, initState)) {
      threadInitState.resultCode = error;
    } else {
      state->isInitialized = true;
    }
  }
  return threadInitState;
}

// Version Management.

cudaError_t cudaDriverGetVersion(int *driverVersion) {
  RETURN_CONVERTED(cuDriverGetVersion(driverVersion));
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
  if (!runtimeVersion) {
    return cudaErrorInvalidValue;
  }
  *runtimeVersion = CUDA_VERSION;
  return cudaSuccess;
}

// Device Management.

// Converts a cudaDeviceAttr to its corresponding CUdevice_attribute.
static CUdevice_attribute convertToDriverDeviceAttribute(
    cudaDeviceAttr attribute) {
  // In CUDA 8.0 and earlier the numeric values of these enums match exactly.
  return static_cast<CUdevice_attribute>(attribute);
}

cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr,
                                   int device) {
  ThreadInitializationState threadInitState = initializeThreadIfNeeded();
  RETURN_CONVERTED_IF_FAIL(threadInitState.resultCode);
  CUdevice_attribute driverAttribute = convertToDriverDeviceAttribute(attr);
  RETURN_CONVERTED(cuDeviceGetAttribute(value, driverAttribute, device));
}

cudaError_t cudaDeviceReset() {
  ThreadInitializationState threadInitState = initializeThreadIfNeeded();
  RETURN_CONVERTED_IF_FAIL(threadInitState.resultCode);
  threadState.isInitialized = false;
  RETURN_CONVERTED(cuDevicePrimaryCtxReset(threadState.activeDeviceOrdinal));
}

cudaError_t cudaDeviceSynchronize() {
  RETURN_CONVERTED_IF_FAIL(initializeIfNeeded()->getResultCode());
  // The CUDA runtime library only understands one context per device, so we can
  // synchronize the current device by synchronizing the current context.
  RETURN_CONVERTED(cuCtxSynchronize());
}

cudaError_t cudaGetDeviceCount(int *count) {
  RETURN_CONVERTED_IF_FAIL(initializeIfNeeded()->getResultCode());
  RETURN_CONVERTED(cuDeviceGetCount(count));
}

cudaError_t cudaSetDevice(int device) {
  auto *initState = initializeIfNeeded();
  RETURN_CONVERTED_IF_FAIL(initState->getResultCode());
  if (!initState->IsValidDevice(device)) {
    RETURN_CUDART(cudaErrorInvalidDevice);
  }
  threadState.activeDeviceOrdinal = device;
  if (threadState.isInitialized) {
    InitializationState *initState = initializeIfNeeded();
    RETURN_CONVERTED_IF_FAIL(setDeviceForThread(device, initState));
  }
  return threadState.lastError;
}

cudaError_t cudaGetDevice(int *device) {
  *device = threadState.activeDeviceOrdinal;
  return threadState.lastError;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
  ThreadInitializationState threadInitState = initializeThreadIfNeeded();
  RETURN_CONVERTED_IF_FAIL(threadInitState.resultCode);
  RETURN_CONVERTED_IF_FAIL(
      cuDeviceGetName(prop->name, sizeof(prop->name) - 1, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceTotalMem(&prop->totalGlobalMem, device));
  int value;
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
  prop->sharedMemPerBlock = value;
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
  RETURN_CONVERTED_IF_FAIL(
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_PITCH, device));
  prop->memPitch = value;
  RETURN_CONVERTED_IF_FAIL(
      cuDeviceGetAttribute(&prop->maxThreadsPerBlock,
                           CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxThreadsDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxThreadsDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxThreadsDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxGridSize[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxGridSize[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxGridSize[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device));
  prop->totalConstMem = value;
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, device));
  prop->textureAlignment = value;
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, device));
  prop->texturePitchAlignment = value;
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->deviceOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, device));
  RETURN_CONVERTED_IF_FAIL(
      cuDeviceGetAttribute(&prop->multiProcessorCount,
                           CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  RETURN_CONVERTED_IF_FAIL(
      cuDeviceGetAttribute(&prop->kernelExecTimeoutEnabled,
                           CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device));
  RETURN_CONVERTED_IF_FAIL(
      cuDeviceGetAttribute(&prop->canMapHostMemory,
                           CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture1D, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture1DMipmap,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture1DLinear,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2D[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2D[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2DMipmap[0],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2DMipmap[1],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2DLinear[0],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2DLinear[1],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2DLinear[2],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2DGather[0],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2DGather[1],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture3D[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture3D[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture3D[2], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture3DAlt[0],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture3DAlt[1],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture3DAlt[2],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTextureCubemap,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture1DLayered[0],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture1DLayered[1],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2DLayered[0],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2DLayered[1],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTexture2DLayered[2],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTextureCubemapLayered[0],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxTextureCubemapLayered[1],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurface1D, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurface2D[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurface2D[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurface3D[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurface3D[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurface3D[2], CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurface1DLayered[0],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurface1DLayered[1],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurface2DLayered[0],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurface2DLayered[1],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurface2DLayered[2],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurfaceCubemap,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurfaceCubemapLayered[0],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxSurfaceCubemapLayered[1],
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
      device));
  prop->surfaceAlignment = value;
  RETURN_CONVERTED_IF_FAIL(
      cuDeviceGetAttribute(&prop->concurrentKernels,
                           CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->ECCEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->pciBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->pciDomainID, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->asyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, device));
  RETURN_CONVERTED_IF_FAIL(
      cuDeviceGetAttribute(&prop->unifiedAddressing,
                           CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->memoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
      device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->l2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->maxThreadsPerMultiProcessor,
      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->streamPrioritiesSupported,
      CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->globalL1CacheSupported,
      CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->localL1CacheSupported,
      CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
      device));
  prop->sharedMemPerMultiprocessor = value;
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->regsPerMultiprocessor,
      CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->managedMemory, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->isMultiGpuBoard, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->multiGpuBoardGroupID, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
      device));
#if CUDA_VERSION >= 8000
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->hostNativeAtomicSupported,
      CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->singleToDoublePrecisionPerfRatio,
      CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, device));
  RETURN_CONVERTED_IF_FAIL(
      cuDeviceGetAttribute(&prop->pageableMemoryAccess,
                           CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device));
  RETURN_CONVERTED_IF_FAIL(cuDeviceGetAttribute(
      &prop->concurrentManagedAccess,
      CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device));
#endif
  return cudaSuccess;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
  RETURN_CONVERTED(
      cuCtxGetStreamPriorityRange(leastPriority, greatestPriority));
}

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
  RETURN_CONVERTED_IF_FAIL(initializeIfNeeded()->getResultCode());
  CUdevice dev = static_cast<CUdevice>(threadState.activeDeviceOrdinal);
  // The numerical values of the flags in the runtime API match the values in
  // the driver API, so we can pass them straight through without modification.
  //
  // See also cudaGetDeviceFlags.
  CUresult result = cuDevicePrimaryCtxSetFlags(dev, flags);
  if (result == CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE) {
    RETURN_CUDART(cudaErrorSetOnActiveProcess);
  }
  RETURN_CONVERTED(result);
}

cudaError_t cudaGetDeviceFlags(unsigned int *flags) {
  // Just as in cudaSetDeviceFlags, the flag values can be passed through
  // without modification.
  //
  // See also cudaSetDeviceFlags.
  RETURN_CONVERTED(cuCtxGetFlags(flags));
}

// Thread Management [DEPRECATED].

cudaError_t cudaThreadSynchronize() { return cudaDeviceSynchronize(); }

// Error Handling.

const char *cudaGetErrorName(cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return "cudaSuccess";
    case cudaErrorMissingConfiguration:
      return "cudaErrorMissingConfiguration";
    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:
      return "cudaErrorInitializationError";
    case cudaErrorLaunchFailure:
      return "cudaErrorLaunchFailure";
    case cudaErrorPriorLaunchFailure:
      return "cudaErrorPriorLaunchFailure";
    case cudaErrorLaunchTimeout:
      return "cudaErrorLaunchTimeout";
    case cudaErrorLaunchOutOfResources:
      return "cudaErrorLaunchOutOfResources";
    case cudaErrorInvalidDeviceFunction:
      return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidConfiguration:
      return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice:
      return "cudaErrorInvalidDevice";
    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";
    case cudaErrorInvalidPitchValue:
      return "cudaErrorInvalidPitchValue";
    case cudaErrorInvalidSymbol:
      return "cudaErrorInvalidSymbol";
    case cudaErrorMapBufferObjectFailed:
      return "cudaErrorMapBufferObjectFailed";
    case cudaErrorUnmapBufferObjectFailed:
      return "cudaErrorUnmapBufferObjectFailed";
    case cudaErrorInvalidHostPointer:
      return "cudaErrorInvalidHostPointer";
    case cudaErrorInvalidDevicePointer:
      return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidTexture:
      return "cudaErrorInvalidTexture";
    case cudaErrorInvalidTextureBinding:
      return "cudaErrorInvalidTextureBinding";
    case cudaErrorInvalidChannelDescriptor:
      return "cudaErrorInvalidChannelDescriptor";
    case cudaErrorInvalidMemcpyDirection:
      return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorAddressOfConstant:
      return "cudaErrorAddressOfConstant";
    case cudaErrorTextureFetchFailed:
      return "cudaErrorTextureFetchFailed";
    case cudaErrorTextureNotBound:
      return "cudaErrorTextureNotBound";
    case cudaErrorSynchronizationError:
      return "cudaErrorSynchronizationError";
    case cudaErrorInvalidFilterSetting:
      return "cudaErrorInvalidFilterSetting";
    case cudaErrorInvalidNormSetting:
      return "cudaErrorInvalidNormSetting";
    case cudaErrorMixedDeviceExecution:
      return "cudaErrorMixedDeviceExecution";
    case cudaErrorCudartUnloading:
      return "cudaErrorCudartUnloading";
    case cudaErrorUnknown:
      return "cudaErrorUnknown";
    case cudaErrorNotYetImplemented:
      return "cudaErrorNotYetImplemented";
    case cudaErrorMemoryValueTooLarge:
      return "cudaErrorMemoryValueTooLarge";
    case cudaErrorInvalidResourceHandle:
      return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady:
      return "cudaErrorNotReady";
    case cudaErrorInsufficientDriver:
      return "cudaErrorInsufficientDriver";
    case cudaErrorSetOnActiveProcess:
      return "cudaErrorSetOnActiveProcess";
    case cudaErrorInvalidSurface:
      return "cudaErrorInvalidSurface";
    case cudaErrorNoDevice:
      return "cudaErrorNoDevice";
    case cudaErrorECCUncorrectable:
      return "cudaErrorECCUncorrectable";
    case cudaErrorSharedObjectSymbolNotFound:
      return "cudaErrorSharedObjectSymbolNotFound";
    case cudaErrorSharedObjectInitFailed:
      return "cudaErrorSharedObjectInitFailed";
    case cudaErrorUnsupportedLimit:
      return "cudaErrorUnsupportedLimit";
    case cudaErrorDuplicateVariableName:
      return "cudaErrorDuplicateVariableName";
    case cudaErrorDuplicateTextureName:
      return "cudaErrorDuplicateTextureName";
    case cudaErrorDuplicateSurfaceName:
      return "cudaErrorDuplicateSurfaceName";
    case cudaErrorDevicesUnavailable:
      return "cudaErrorDevicesUnavailable";
    case cudaErrorInvalidKernelImage:
      return "cudaErrorInvalidKernelImage";
    case cudaErrorNoKernelImageForDevice:
      return "cudaErrorNoKernelImageForDevice";
    case cudaErrorIncompatibleDriverContext:
      return "cudaErrorIncompatibleDriverContext";
    case cudaErrorPeerAccessAlreadyEnabled:
      return "cudaErrorPeerAccessAlreadyEnabled";
    case cudaErrorPeerAccessNotEnabled:
      return "cudaErrorPeerAccessNotEnabled";
    case cudaErrorDeviceAlreadyInUse:
      return "cudaErrorDeviceAlreadyInUse";
    case cudaErrorProfilerDisabled:
      return "cudaErrorProfilerDisabled";
    case cudaErrorProfilerNotInitialized:
      return "cudaErrorProfilerNotInitialized";
    case cudaErrorProfilerAlreadyStarted:
      return "cudaErrorProfilerAlreadyStarted";
    case cudaErrorProfilerAlreadyStopped:
      return "cudaErrorProfilerAlreadyStopped";
    case cudaErrorAssert:
      return "cudaErrorAssert";
    case cudaErrorTooManyPeers:
      return "cudaErrorTooManyPeers";
    case cudaErrorHostMemoryAlreadyRegistered:
      return "cudaErrorHostMemoryAlreadyRegistered";
    case cudaErrorHostMemoryNotRegistered:
      return "cudaErrorHostMemoryNotRegistered";
    case cudaErrorOperatingSystem:
      return "cudaErrorOperatingSystem";
    case cudaErrorPeerAccessUnsupported:
      return "cudaErrorPeerAccessUnsupported";
    case cudaErrorLaunchMaxDepthExceeded:
      return "cudaErrorLaunchMaxDepthExceeded";
    case cudaErrorLaunchFileScopedTex:
      return "cudaErrorLaunchFileScopedTex";
    case cudaErrorLaunchFileScopedSurf:
      return "cudaErrorLaunchFileScopedSurf";
    case cudaErrorSyncDepthExceeded:
      return "cudaErrorSyncDepthExceeded";
    case cudaErrorLaunchPendingCountExceeded:
      return "cudaErrorLaunchPendingCountExceeded";
    case cudaErrorNotPermitted:
      return "cudaErrorNotPermitted";
    case cudaErrorNotSupported:
      return "cudaErrorNotSupported";
    case cudaErrorHardwareStackError:
      return "cudaErrorHardwareStackError";
    case cudaErrorIllegalInstruction:
      return "cudaErrorIllegalInstruction";
    case cudaErrorMisalignedAddress:
      return "cudaErrorMisalignedAddress";
    case cudaErrorInvalidAddressSpace:
      return "cudaErrorInvalidAddressSpace";
    case cudaErrorInvalidPc:
      return "cudaErrorInvalidPc";
    case cudaErrorIllegalAddress:
      return "cudaErrorIllegalAddress";
    case cudaErrorInvalidPtx:
      return "cudaErrorInvalidPtx";
    case cudaErrorInvalidGraphicsContext:
      return "cudaErrorInvalidGraphicsContext";
    case cudaErrorStartupFailure:
      return "cudaErrorStartupFailure";
    case cudaErrorApiFailureBase:
      return "cudaErrorApiFailureBase";
    default:
      return "UNKNOWN CUDA ERROR";
  }
}

const char *cudaGetErrorString(cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return "no error";
    case cudaErrorMissingConfiguration:
      return "__global__ function call is not configured";
    case cudaErrorMemoryAllocation:
      return "out of memory";
    case cudaErrorInitializationError:
      return "initialization error";
    case cudaErrorLaunchFailure:
      return "unspecified launch failure";
    case cudaErrorPriorLaunchFailure:
      return "unspecified launch failure in prior launch";
    case cudaErrorLaunchTimeout:
      return "the launch timed out and was terminated";
    case cudaErrorLaunchOutOfResources:
      return "too many resources requested for launch";
    case cudaErrorInvalidDeviceFunction:
      return "invalid device function";
    case cudaErrorInvalidConfiguration:
      return "invalid configuration argument";
    case cudaErrorInvalidDevice:
      return "invalid device ordinal";
    case cudaErrorInvalidValue:
      return "invalid argument";
    case cudaErrorInvalidPitchValue:
      return "invalid pitch argument";
    case cudaErrorInvalidSymbol:
      return "invalid device symbol";
    case cudaErrorMapBufferObjectFailed:
      return "mapping of buffer object failed";
    case cudaErrorUnmapBufferObjectFailed:
      return "unmapping of buffer object failed";
    case cudaErrorInvalidHostPointer:
      return "invalid host pointer";
    case cudaErrorInvalidDevicePointer:
      return "invalid device pointer";
    case cudaErrorInvalidTexture:
      return "invalid texture reference";
    case cudaErrorInvalidTextureBinding:
      return "texture is not bound to a pointer";
    case cudaErrorInvalidChannelDescriptor:
      return "invalid channel descriptor";
    case cudaErrorInvalidMemcpyDirection:
      return "invalid copy direction for memcpy";
    case cudaErrorAddressOfConstant:
      return "invalid address of constant";
    case cudaErrorTextureFetchFailed:
      return "fetch from texture failed";
    case cudaErrorTextureNotBound:
      return "cannot fetch from a texture that is not bound";
    case cudaErrorSynchronizationError:
      return "incorrect use of __syncthreads()";
    case cudaErrorInvalidFilterSetting:
      return "linear filtering not supported for non-float type";
    case cudaErrorInvalidNormSetting:
      return "read as normalized float not supported for 32-bit non float type";
    case cudaErrorMixedDeviceExecution:
      return "device emulation mode and device execution mode cannot be mixed";
    case cudaErrorCudartUnloading:
      return "driver shutting down";
    case cudaErrorUnknown:
      return "unknown error";
    case cudaErrorNotYetImplemented:
      return "feature not yet implemented";
    case cudaErrorMemoryValueTooLarge:
      return "memory size or pointer value too large to fit in 32 bit";
    case cudaErrorInvalidResourceHandle:
      return "invalid resource handle";
    case cudaErrorNotReady:
      return "device not ready";
    case cudaErrorInsufficientDriver:
      return "CUDA driver version is insufficient for CUDA runtime version";
    case cudaErrorSetOnActiveProcess:
      return "cannot set while device is active in this process";
    case cudaErrorInvalidSurface:
      return "invalid surface reference";
    case cudaErrorNoDevice:
      return "no CUDA-capable device is detected";
    case cudaErrorECCUncorrectable:
      return "uncorrectable ECC error encountered";
    case cudaErrorSharedObjectSymbolNotFound:
      return "shared object symbol not found";
    case cudaErrorSharedObjectInitFailed:
      return "shared object initialization failed";
    case cudaErrorUnsupportedLimit:
      return "limit is not supported on this architecture";
    case cudaErrorDuplicateVariableName:
      return "duplicate global variable looked up by string name";
    case cudaErrorDuplicateTextureName:
      return "duplicate texture looked up by string name";
    case cudaErrorDuplicateSurfaceName:
      return "duplicate surface looked up by string name";
    case cudaErrorDevicesUnavailable:
      return "all CUDA-capable devices are busy or unavailable";
    case cudaErrorInvalidKernelImage:
      return "device kernel image is invalid";
    case cudaErrorNoKernelImageForDevice:
      return "no kernel image is available for execution on the device";
    case cudaErrorIncompatibleDriverContext:
      return "incompatible driver context";
    case cudaErrorPeerAccessAlreadyEnabled:
      return "peer access is already enabled";
    case cudaErrorPeerAccessNotEnabled:
      return "peer access has not been enabled";
    case cudaErrorDeviceAlreadyInUse:
      return "exclusive-thread device already in use by a different thread";
    case cudaErrorProfilerDisabled:
      return "profiler disabled while using external profiling tool";
    case cudaErrorProfilerNotInitialized:
      return "profiler not initialized: call cudaProfilerInitialize()";
    case cudaErrorProfilerAlreadyStarted:
      return "profiler already started";
    case cudaErrorProfilerAlreadyStopped:
      return "profiler already stopped";
    case cudaErrorAssert:
      return "device-side assert triggered";
    case cudaErrorTooManyPeers:
      return "peer mapping resources exhausted";
    case cudaErrorHostMemoryAlreadyRegistered:
      return "part or all of the requested memory range is already mapped";
    case cudaErrorHostMemoryNotRegistered:
      return "pointer does not correspond to a registered memory region";
    case cudaErrorOperatingSystem:
      return "OS call failed or operation not supported on this OS";
    case cudaErrorPeerAccessUnsupported:
      return "peer access is not supported between these two devices";
    case cudaErrorLaunchMaxDepthExceeded:
      return "launch would exceed maximum depth of nested launches";
    case cudaErrorLaunchFileScopedTex:
      return "launch failed because kernel uses unsupported, file-scoped "
             "textures (texture objects are supported)";
    case cudaErrorLaunchFileScopedSurf:
      return "launch failed because kernel uses unsupported, file-scoped "
             "surfaces (surface objects are supported)";
    case cudaErrorSyncDepthExceeded:
      return "cudaDeviceSynchronize failed because caller's grid depth exceeds "
             "cudaLimitDevRuntimeSyncDepth";
    case cudaErrorLaunchPendingCountExceeded:
      return "launch failed because launch would exceed "
             "cudaLimitDevRuntimePendingLaunchCount";
    case cudaErrorNotPermitted:
      return "operation not permitted";
    case cudaErrorNotSupported:
      return "operation not supported";
    case cudaErrorHardwareStackError:
      return "hardware stack error";
    case cudaErrorIllegalInstruction:
      return "an illegal instruction was encountered";
    case cudaErrorMisalignedAddress:
      return "misaligned address";
    case cudaErrorInvalidAddressSpace:
      return "operation not supported on global/shared address space";
    case cudaErrorInvalidPc:
      return "invalid program counter";
    case cudaErrorIllegalAddress:
      return "an illegal memory access was encountered";
    case cudaErrorInvalidPtx:
      return "a PTX JIT compilation failed";
    case cudaErrorInvalidGraphicsContext:
      return "invalid OpenGL or DirectX context";
    case cudaErrorStartupFailure:
      return "startup failure in cuda runtime";
    case cudaErrorApiFailureBase:
      return "api failure base";
    default:
      return "UNKNOWN CUDA ERROR";
  }
}

cudaError_t cudaGetLastError() {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  cudaError_t lastError = threadState.lastError;
  threadState.lastError = cudaSuccess;
  return lastError;
}

cudaError_t cudaPeekAtLastError() {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  return threadState.lastError;
}

// Stream Management.

namespace {

// The user data passed to the driver callback function.
struct CudartCallbackData {
  cudaStreamCallback_t callback;
  void *userData;
};

}  // namespace

// Interprets the userData as a CudartCallbackData struct and calls the callback
// stored there.
//
// Takes ownership of the data pointed to by userData.
static void CUDA_CB driverCallback(CUstream hStream, CUresult status,
                                   void *userData) {
  auto cudartCallbackData = std::unique_ptr<CudartCallbackData>(
      static_cast<CudartCallbackData *>(userData));
  cudaError_t cudartStatus = convertToCudartError(status);
  cudartCallbackData->callback(hStream, cudartStatus,
                               cudartCallbackData->userData);
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback, void *userData,
                                  unsigned int flags) {
  std::unique_ptr<CudartCallbackData> data(new CudartCallbackData);
  data->callback = callback;
  data->userData = userData;
  RETURN_CONVERTED(
      cuStreamAddCallback(stream, driverCallback, data.release(), flags));
}

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuStreamCreate(pStream, CU_STREAM_DEFAULT));
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                      unsigned int flags) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuStreamCreate(pStream, flags));
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                         unsigned int flags, int priority) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuStreamCreateWithPriority(pStream, flags, priority));
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuStreamDestroy(stream));
}

cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuStreamGetFlags(hStream, flags));
}

cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuStreamGetPriority(hStream, priority));
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuStreamQuery(stream));
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuStreamSynchronize(stream));
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuStreamWaitEvent(stream, event, flags));
}

// Event Management.

cudaError_t cudaEventCreate(cudaEvent_t *event) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuEventCreate(event, CU_EVENT_DEFAULT));
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuEventCreate(event, flags));
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuEventDestroy(event));
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                 cudaEvent_t end) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuEventElapsedTime(ms, start, end));
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  // CUDA_ERROR_NOT_READY is not a real error in this case.
  CUresult result = cuEventQuery(event);
  if (result == CUDA_SUCCESS || result == CUDA_ERROR_NOT_READY) {
    return convertToCudartError(result);
  }
  RETURN_CONVERTED(result);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuEventRecord(event, stream));
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuEventSynchronize(event));
}

// Device code registration.

namespace {

std::string MangleNamespaceNestedName(absl::string_view name) {
  // Drop leading "::" if present -- we assume this name is in the global
  // namespace either way.
  absl::ConsumePrefix(&name, "::");
  if (!absl::StrContains(name, "::")) {
    return std::string(name);
  }
  std::string ret = "_ZN";
  for (absl::string_view component : absl::StrSplit(name, "::")) {
    absl::StrAppend(&ret, component.size(), component);
  }
  absl::StrAppend(&ret, "E");
  return ret;
}

// Information for a symbol stored in device memory.
struct DeviceSymbolInfo {
  DeviceSymbolInfo() : dptr(0), size(0) {}
  DeviceSymbolInfo(CUdeviceptr dptr, size_t size) : dptr(dptr), size(size) {}
  DeviceSymbolInfo(const DeviceSymbolInfo &) = default;

  // Device pointer to the symbol.
  CUdeviceptr dptr;

  // Number of bytes used to store the symbol contents.
  size_t size;
};

// Struct used to pass arguments to __cudaRegisterFatBinary.
struct CudaRegisterFatBinaryArguments {
  int magic_number;
  int version;
  const void *binary;
  void *unused;
};

// The binary code for a module and handles to all the contexts in which it is
// loaded.
//
// Modules are lazily loaded into contexts as needed because some CUDA SDK
// libraries register modules which cannot actually be loaded on all GPUs.
class Module {
 public:
  // Creates a module that is not yet loaded in any context.
  explicit Module(const void *binary) : binary_(binary) {}

  // Unloads the module from each context into which it has been loaded.
  ~Module() {
    MutexLock Lock(mutex_);
    for (auto pair : loadedModules_) {
      cuModuleUnload(pair.second);
    }
  }

  // Gets a driver handle to a loaded module for the currently active context.
  //
  // This will load the module in the currently active context if it is not
  // already loaded there.
  //
  // Returns nullptr if the module cannot be loaded.
  //
  // This function is thread-safe.
  CUmodule GetDriverHandleForActiveContext() {
    if (CUresult result = initializeThreadIfNeeded().resultCode) {
      std::cerr << "Failed to initialize thread for CUDA. "
                   << GetErrorDescription(result);
      return nullptr;
    }
    CUcontext context = threadState.activeContext;
    MutexLock Lock(mutex_);
    auto iterator = loadedModules_.find(context);
    if (iterator != loadedModules_.end()) {
      return iterator->second;
    }
    CUmodule module;
    if (CUresult result = cuModuleLoadFatBinary(&module, binary_)) {
      std::cerr << "Failed to load CUDA fat binary. "
                << GetErrorDescription(result);
      return nullptr;
    }
    loadedModules_.emplace(context, module);
    return module;
  }

 private:
  std::mutex mutex_;

  // Binary data used to create the module.
  const void *binary_;

  // Loaded instances of the module.
  std::map<CUcontext, CUmodule> loadedModules_; //  ABSL_GUARDED_BY(mutex_);
};

// A class to track all registered modules.
//
// The functions of this class are thread-safe.
class ModuleRegistry {
 public:
  // Registers a module for the given binary and returns a raw pointer to the
  // created Module object.
  //
  // Ownership of the returned Module is not passed to the caller, it is
  // retained by the registry.
  Module *RegisterModule(const void *binary) {
    MutexLock Lock(mutex_);
    std::unique_ptr<Module> pointer(new Module(binary));
    auto rawPointer = pointer.get();
    modules_.emplace(rawPointer, std::move(pointer));
    return rawPointer;
  }

  // Unregisters a module created by a prior call to RegisterModule.
  void UnregisterModule(Module *handle) {
    MutexLock Lock(mutex_);
    modules_.erase(handle);
  }

 private:
  std::mutex mutex_;
  std::map<Module *, std::unique_ptr<Module>> modules_;// ABSL_GUARDED_BY(mutex_);
};

// A named object that is defined in a given CUDA module.
//
// Examples of types of module objects are kernel functions, global variables,
// and texture references.
//
// DriverHandle is the type of handle used by the driver API to refer to the
// object from host code. For example, kernel functions use CUfunction, texture
// references use CUtexref, and global variables use DeviceSymbolInfo, which
// contains a CUdeviceptr and a size.
//
// This abstract base class holds the code common to all types of module
// objects. A subclass will be created for each specific module object type.
//
// Functions are thread-safe, and implementations of the pure virtual methods
// must also be thread-safe.
template <typename DriverHandle>
class ModuleObject {
 public:
  // Creates an object with a given name in the given module.
  ModuleObject(Module *module, std::string name)
      : name(name), module_(module) {}

  virtual ~ModuleObject() = default;

  // Gets a driver host handle to the object for the current context.
  //
  // If the module for this object has not been loaded in the current context,
  // this function will first load the module.
  //
  // Returns a null DriverHandle if a handle cannot be obtained.
  DriverHandle GetDriverHandleForActiveContext() {
    if (CUresult result = initializeThreadIfNeeded().resultCode) {
      std::cerr << "Failed to initialize thread for CUDA. "
                << GetErrorDescription(result);
      return MakeNull();
    }
    CUcontext context = threadState.activeContext;

    MutexLock Lock(mutex_);
    auto iterator = values_.find(context);
    if (iterator != values_.end()) {
      return iterator->second;
    }
    CUmodule driverModule = module_->GetDriverHandleForActiveContext();
    if (!driverModule) {
      return MakeNull();
    }
    DriverHandle handle = GetDriverHandleForDriverModule(driverModule);
    if (!IsNull(handle)) {
      values_.emplace(context, handle);
    }
    return handle;
  }

 protected:
  // Gets the driver handle for this object for the given driver module handle.
  //
  // This function is called after the module for this object has already been
  // loaded in the relevant context and the driver handle for that module in
  // that context is passed as the argument here.
  //
  // Returns a null DriverHandle if a handle cannot be obtained.
  virtual DriverHandle GetDriverHandleForDriverModule(
      CUmodule driverModule) = 0;

  // Makes a DriverHandle with a null value.
  virtual DriverHandle MakeNull() = 0;

  // Returns true if the argument has a null value.
  virtual bool IsNull(DriverHandle handle) = 0;

  const std::string name;

 private:
  std::mutex mutex_;
  Module *module_; // ABSL_GUARDED_BY(mutex_);
  std::map<CUcontext, DriverHandle> values_; // ABSL_GUARDED_BY(mutex_);
};

class KernelObject : public ModuleObject<CUfunction> {
 public:
  KernelObject(Module *module, const char *name)
      : ModuleObject<CUfunction>(module, name) {}

 protected:
  CUfunction GetDriverHandleForDriverModule(CUmodule driverModule) override {
    CUfunction function;
    CUresult error = cuModuleGetFunction(&function, driverModule, name.c_str());
    if (error) {
      std::cerr << "Failed to get function from module: function name = "
                << name << ", " << GetErrorDescription(error);
      return nullptr;
    }
    return function;
  }

  CUfunction MakeNull() override { return nullptr; }
  bool IsNull(CUfunction handle) override { return handle == nullptr; }
};

class DeviceSymbolInfoObject : public ModuleObject<DeviceSymbolInfo> {
 public:
  DeviceSymbolInfoObject(Module *module, const char *name)
      : ModuleObject<DeviceSymbolInfo>(module, name) {}

 protected:
  DeviceSymbolInfo GetDriverHandleForDriverModule(
      CUmodule driverModule) override {
    std::string mangledName = MangleNamespaceNestedName(name);
    CUdeviceptr dptr;
    size_t bytes;
    CUresult error =
        cuModuleGetGlobal(&dptr, &bytes, driverModule, mangledName.c_str());
    if (error) {
      std::cerr << "Failed to get global symbol from module: symbol name = "
                   << name << ", " << GetErrorDescription(error);
      return DeviceSymbolInfo();
    }
    return DeviceSymbolInfo(dptr, bytes);
  }

  DeviceSymbolInfo MakeNull() override { return DeviceSymbolInfo(); }
  bool IsNull(DeviceSymbolInfo handle) override { return handle.dptr == 0; }
};

class TextureObject : public ModuleObject<CUtexref> {
 public:
  TextureObject(Module *module, const char *name)
      : ModuleObject<CUtexref>(module, name) {}

 protected:
  CUtexref GetDriverHandleForDriverModule(CUmodule driverModule) override {
    std::string mangledName = MangleNamespaceNestedName(name);
    CUtexref texref;
    CUresult error =
        cuModuleGetTexRef(&texref, driverModule, mangledName.c_str());
    if (error) {
      std::cerr << "Failed to get texture reference from module: texture "
                      "reference name = "
                   << name << ", " << GetErrorDescription(error);
      return nullptr;
    }
    return texref;
  }

  CUtexref MakeNull() override { return nullptr; }
  bool IsNull(CUtexref handle) override { return handle == nullptr; }
};

// A place to store all the registered ModuleObjects of a given type.
//
// For example, there can be a ModuleObjectRegistry for kernels, another for
// global variables, and another for textures.
//
// The functions of this class are thread-safe.
template <typename ObjectType>
class ModuleObjectRegistry {
 public:
  // Registers the hostHandle to refer to the object of the given name within
  // the given module.
  //
  // Returns true if the registration was successful or false if the given host
  // handle was already registered for another object.
  bool Insert(const void *hostHandle, Module *module, const char *name) {
    MutexLock writeLock(mutex_);
    std::unique_ptr<ObjectType> pointer(new ObjectType(module, name));
    return objects_.emplace(hostHandle, std::move(pointer)).second;
  }

  // Gets a pointer to the object for the given handle.
  //
  // Returns nullptr if there is no object for the given handle.
  //
  // Ownership of the object is NOT passed to the caller, it is retained by the
  // registry.
  ObjectType *Find(const void *hostHandle) {
    MutexLock readerLock(mutex_);
    auto iterator = objects_.find(hostHandle);
    return (iterator == objects_.end()) ? nullptr : iterator->second.get();
  }

 private:
  std::mutex mutex_;
  std::map<const void *, std::unique_ptr<ObjectType>> objects_;
  //ABSL_GUARDED_BY(mutex_);
};

// Registry of modules.
ModuleRegistry& getModuleRegistry() {
  static ModuleRegistry moduleRegistry;
  return moduleRegistry;
}

// Registry of kernels.
ModuleObjectRegistry<KernelObject>& getFunctionRegistry() {
  static ModuleObjectRegistry<KernelObject> functionRegistry;
  return functionRegistry;
}

// Registry of symbols.
ModuleObjectRegistry<DeviceSymbolInfoObject>& getDeviceSymbolRegistry() {
  static ModuleObjectRegistry<DeviceSymbolInfoObject> deviceSymbolRegistry;
  return deviceSymbolRegistry;
}

// Registry of texture references.
ModuleObjectRegistry<TextureObject>& getTextureRegistry() {
  static ModuleObjectRegistry<TextureObject> textureRegistry;
  return textureRegistry;
}

}  // namespace

extern "C" void** __cudaRegisterFatBinary(void* fatCubin) {
  auto *arguments = static_cast<CudaRegisterFatBinaryArguments *>(fatCubin);
  // TODO: Should I check the magic number, version number, etc?
  return reinterpret_cast<void **>(
      getModuleRegistry().RegisterModule(arguments->binary));
}

extern "C" void __cudaUnregisterFatBinary(void** fatCubinHandle) {
  getModuleRegistry().UnregisterModule(
      reinterpret_cast<Module*>(fatCubinHandle));
}

extern "C" void __cudaRegisterFunction(void** fatCubinHandle,
                                       const char* hostFun, char* /*deviceFun*/,
                                       const char* deviceName,
                                       int /*thread_limit*/, uint3* /*tid*/,
                                       uint3* /*bid*/, dim3* /*bDim*/,
                                       dim3* /*gDim*/, int* /*wSize*/) {
  // TODO: What do I do with deviceFun, thread_limit, tid, bid, bDim,
  // gDim, and wSize?
  Module *module = reinterpret_cast<Module *>(fatCubinHandle);
  getFunctionRegistry().Insert(hostFun, module, deviceName);
}

extern "C" void __cudaRegisterVar(void** fatCubinHandle, char* hostVar,
                                  char* /*deviceAddress*/,
                                  const char* deviceName, int /*ext*/,
                                  int /*size*/, int /*constant*/,
                                  int /*global*/) {
  // TODO: What do I do with deviceAddress, ext, constant, and global?
  Module *module = reinterpret_cast<Module *>(fatCubinHandle);
  getDeviceSymbolRegistry().Insert(hostVar, module, deviceName);
}

extern "C" void __cudaRegisterTexture(void** fatCubinHandle,
                                      const struct textureReference* hostVar,
                                      const void** /*deviceAddress*/,
                                      const char* deviceName, int /*dim*/,
                                      int /*norm*/, int /*ext*/
) {
  // TODO: What do I do with deviceAddress, dim, norm, and ext?
  Module *module = reinterpret_cast<Module *>(fatCubinHandle);
  getTextureRegistry().Insert(hostVar, module, deviceName);
}

static CUfunction getCuFunctionFromHandle(const void *func) {
  if (KernelObject *object = getFunctionRegistry().Find(func)) {
    return object->GetDriverHandleForActiveContext();
  }
  return nullptr;
}

static DeviceSymbolInfo getDeviceSymbolInfoFromHandle(const void *symbol) {
  if (DeviceSymbolInfoObject* object = getDeviceSymbolRegistry().Find(symbol)) {
    return object->GetDriverHandleForActiveContext();
  }
  return DeviceSymbolInfo();
}

static CUtexref getDriverTexRefFromHandle(const void *handle) {
  if (TextureObject *object = getTextureRegistry().Find(handle)) {
    return object->GetDriverHandleForActiveContext();
  }
  return nullptr;
}

// Execution control.

cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr,
                                             const void* func) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  CUfunction cuFunction = getCuFunctionFromHandle(func);
  if (!cuFunction) {
    RETURN_CUDART(cudaErrorInvalidDeviceFunction);
  }
  int value;
  RETURN_CONVERTED_IF_FAIL(
      cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_BINARY_VERSION, cuFunction));
  attr->binaryVersion = value;
  RETURN_CONVERTED_IF_FAIL(
      cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, cuFunction));
  attr->cacheModeCA = value;
  RETURN_CONVERTED_IF_FAIL(cuFuncGetAttribute(
      &value, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, cuFunction));
  attr->constSizeBytes = value;
  RETURN_CONVERTED_IF_FAIL(cuFuncGetAttribute(
      &value, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, cuFunction));
  attr->localSizeBytes = value;
  RETURN_CONVERTED_IF_FAIL(cuFuncGetAttribute(
      &value, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cuFunction));
  attr->maxThreadsPerBlock = value;
  RETURN_CONVERTED_IF_FAIL(
      cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_NUM_REGS, cuFunction));
  attr->numRegs = value;
  RETURN_CONVERTED_IF_FAIL(
      cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_PTX_VERSION, cuFunction));
  attr->ptxVersion = value;
  RETURN_CONVERTED_IF_FAIL(cuFuncGetAttribute(
      &value, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, cuFunction));
  attr->sharedSizeBytes = value;
  return cudaSuccess;
}

extern "C" cudaError_t cudaFuncSetAttribute(const void* func,
                                            enum cudaFuncAttribute attr,
                                            int value) {
  // TODO: uniplemented yet.
  return cudaErrorUnknown;
}

cudaError_t cudaFuncSetCacheConfig(const void *func,
                                   cudaFuncCache cacheConfig) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  CUfunction cuFunction = getCuFunctionFromHandle(func);
  if (!cuFunction) {
    RETURN_CUDART(cudaErrorInvalidDeviceFunction);
  }
  CUfunc_cache cuCacheConfig = CU_FUNC_CACHE_PREFER_NONE;
  switch (cacheConfig) {
    case cudaFuncCachePreferNone:
      cuCacheConfig = CU_FUNC_CACHE_PREFER_NONE;
      break;
    case cudaFuncCachePreferShared:
      cuCacheConfig = CU_FUNC_CACHE_PREFER_SHARED;
      break;
    case cudaFuncCachePreferL1:
      cuCacheConfig = CU_FUNC_CACHE_PREFER_L1;
      break;
    case cudaFuncCachePreferEqual:
      cuCacheConfig = CU_FUNC_CACHE_PREFER_EQUAL;
      break;
    default:
      cuCacheConfig = CU_FUNC_CACHE_PREFER_NONE;
      break;
  }
  RETURN_CONVERTED(cuFuncSetCacheConfig(cuFunction, cuCacheConfig));
}

cudaError_t cudaFuncSetSharedMemConfig(const void *func,
                                       cudaSharedMemConfig config) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  CUfunction cuFunction = getCuFunctionFromHandle(func);
  if (!cuFunction) {
    RETURN_CUDART(cudaErrorInvalidDeviceFunction);
  }
  CUsharedconfig cuConfig = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
  switch (config) {
    case cudaSharedMemBankSizeDefault:
      cuConfig = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
      break;
    case cudaSharedMemBankSizeFourByte:
      cuConfig = CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE;
      break;
    case cudaSharedMemBankSizeEightByte:
      cuConfig = CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE;
      break;
    default:
      cuConfig = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
      break;
  }
  RETURN_CONVERTED(cuFuncSetSharedMemConfig(cuFunction, cuConfig));
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  CUfunction cuFunction = getCuFunctionFromHandle(func);
  if (!cuFunction) {
    RETURN_CUDART(cudaErrorInvalidDeviceFunction);
  }
  void **extras = nullptr;
  RETURN_CONVERTED(cuLaunchKernel(cuFunction, gridDim.x, gridDim.y, gridDim.z,
                                  blockDim.x, blockDim.y, blockDim.z, sharedMem,
                                  stream, args, extras));
}

// Deprecated execution control.

namespace {

struct ExecutionStackNode {
  dim3 gridDim;
  dim3 blockDim;
  size_t sharedMem;
  cudaStream_t stream;
  std::vector<char> argumentBuffer;

  ExecutionStackNode(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                     cudaStream_t stream)
      : gridDim(gridDim),
        blockDim(blockDim),
        sharedMem(sharedMem),
        stream(stream) {}
};

thread_local std::vector<ExecutionStackNode>  executionStack;

}  // namespace

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                              cudaStream_t stream) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  executionStack.emplace_back(gridDim, blockDim, sharedMem, stream);
  return cudaSuccess;
}

cudaError_t cudaLaunch(const void *func) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  std::vector<ExecutionStackNode> *stack = &executionStack;
  if (stack->empty()) {
    RETURN_CUDART(cudaErrorInvalidConfiguration);
  }
  CUfunction cuFunction = getCuFunctionFromHandle(func);
  if (!cuFunction) {
    RETURN_CUDART(cudaErrorInvalidDeviceFunction);
  }
  ExecutionStackNode node = std::move(stack->back());
  stack->pop_back();
  size_t argumentBufferSize = node.argumentBuffer.size();
  std::array<void *, 5> extra = {
      {CU_LAUNCH_PARAM_BUFFER_POINTER, node.argumentBuffer.data(),
       CU_LAUNCH_PARAM_BUFFER_SIZE, &argumentBufferSize, CU_LAUNCH_PARAM_END}};
  RETURN_CONVERTED(
      cuLaunchKernel(cuFunction, node.gridDim.x, node.gridDim.y, node.gridDim.z,
                     node.blockDim.x, node.blockDim.y, node.blockDim.z,
                     node.sharedMem, node.stream, nullptr, extra.data()));
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  std::vector<ExecutionStackNode> *stack = &executionStack;
  if (stack->empty()) {
    RETURN_CUDART(cudaErrorInvalidConfiguration);
  }
  auto &argumentBuffer = stack->back().argumentBuffer;
  if (offset + size > argumentBuffer.size()) {
    argumentBuffer.resize(offset + size);
  }
  std::memcpy(argumentBuffer.data() + offset, arg, size);
  return cudaSuccess;
}

// Device memory management.

static CUdeviceptr convertToCuDevicePtr(const void *devPtr) {
  static_assert(sizeof(void *) == sizeof(CUdeviceptr),
                "CUdeviceptr size does not equal void * pointer size");
  CUdeviceptr result;
  std::memcpy(&result, &devPtr, sizeof(CUdeviceptr));
  return result;
}

cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc *desc, cudaExtent *extent,
                             unsigned int *flags, cudaArray_t array) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  CUDA_ARRAY_DESCRIPTOR arrayDescriptor;
  RETURN_CONVERTED_IF_FAIL(
      cuArrayGetDescriptor(&arrayDescriptor, reinterpret_cast<CUarray>(array)));
  int bitCount = 0;
  switch (arrayDescriptor.Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
      desc->f = cudaChannelFormatKindUnsigned;
      bitCount = 8;
      break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
      desc->f = cudaChannelFormatKindUnsigned;
      bitCount = 16;
      break;
    case CU_AD_FORMAT_UNSIGNED_INT32:
      desc->f = cudaChannelFormatKindUnsigned;
      bitCount = 32;
      break;
    case CU_AD_FORMAT_SIGNED_INT8:
      desc->f = cudaChannelFormatKindSigned;
      bitCount = 8;
      break;
    case CU_AD_FORMAT_SIGNED_INT16:
      desc->f = cudaChannelFormatKindSigned;
      bitCount = 16;
      break;
    case CU_AD_FORMAT_SIGNED_INT32:
      desc->f = cudaChannelFormatKindSigned;
      bitCount = 32;
      break;
    case CU_AD_FORMAT_HALF:
      desc->f = cudaChannelFormatKindFloat;
      bitCount = 16;
      break;
    case CU_AD_FORMAT_FLOAT:
      desc->f = cudaChannelFormatKindFloat;
      bitCount = 32;
      break;
  }
  desc->x = 0;
  desc->y = 0;
  desc->z = 0;
  desc->w = 0;
  switch (arrayDescriptor.NumChannels) {
    case 4:
      desc->w = bitCount;
      //ABSL_FALLTHROUGH_INTENDED;
    case 3:
      desc->z = bitCount;
      //ABSL_FALLTHROUGH_INTENDED;
    case 2:
      desc->y = bitCount;
      //ABSL_FALLTHROUGH_INTENDED;
    case 1:
      desc->x = bitCount;
  }

  extent->width = arrayDescriptor.Width;
  extent->height = arrayDescriptor.Height;
  extent->depth = 0;

  // TODO: What about flags?
  *flags = 0;

  return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuMemFree(convertToCuDevicePtr(devPtr)));
}

cudaError_t cudaFreeHost(void *ptr) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuMemFreeHost(ptr));
}

cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  DeviceSymbolInfo info = getDeviceSymbolInfoFromHandle(symbol);
  if (!info.dptr) {
    RETURN_CUDART(cudaErrorInvalidSymbol);
  }
  *devPtr = reinterpret_cast<void *>(info.dptr);
  return cudaSuccess;
}

cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  DeviceSymbolInfo info = getDeviceSymbolInfoFromHandle(symbol);
  if (!info.dptr) {
    RETURN_CUDART(cudaErrorInvalidSymbol);
  }
  *size = info.size;
  return cudaSuccess;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuMemHostAlloc(pHost, size, flags));
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuMemHostRegister(ptr, size, flags));
}

cudaError_t cudaHostUnregister(void *ptr) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuMemHostUnregister(ptr));
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuMemAllocHost(ptr, size));
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuMemAlloc(reinterpret_cast<CUdeviceptr *>(devPtr), size));
}

static CUarray_format getCuArrayFormat(
    const cudaChannelFormatDesc &channelDesc) {
  switch (channelDesc.f) {
    case cudaChannelFormatKindSigned:
      return CU_AD_FORMAT_SIGNED_INT32;
    case cudaChannelFormatKindUnsigned:
      return CU_AD_FORMAT_UNSIGNED_INT32;
    case cudaChannelFormatKindFloat:
      return CU_AD_FORMAT_FLOAT;
    case cudaChannelFormatKindNone:
      return CU_AD_FORMAT_FLOAT;
  }
}

static unsigned int getNumChannels(const cudaChannelFormatDesc &channelDesc) {
  if (channelDesc.w) {
    return 4;
  } else if (channelDesc.z) {
    return 3;
  } else if (channelDesc.y) {
    return 2;
  } else if (channelDesc.x) {
    return 1;
  } else {
    return 0;
  }
}

cudaError_t cudaFreeArray(cudaArray_t array) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuArrayDestroy(reinterpret_cast<CUarray>(array)));
}

cudaError_t cudaMallocArray(cudaArray_t *array,
                            const cudaChannelFormatDesc *desc, size_t width,
                            size_t height, unsigned int flags) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
  arrayDesc.Width = width;
  arrayDesc.Height = height;
  arrayDesc.Depth = 0;
  arrayDesc.Format = getCuArrayFormat(*desc);
  arrayDesc.NumChannels = getNumChannels(*desc);
  arrayDesc.Flags =
      ((flags & cudaArraySurfaceLoadStore) ? CUDA_ARRAY3D_SURFACE_LDST : 0u) |
      ((flags & cudaArrayTextureGather) ? CUDA_ARRAY3D_TEXTURE_GATHER : 0u);
  RETURN_CONVERTED(
      cuArray3DCreate(reinterpret_cast<CUarray *>(array), &arrayDesc));
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count,
                                 size_t offset, cudaMemcpyKind kind) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  if (kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyDeviceToDevice) {
    RETURN_CUDART(cudaErrorInvalidMemcpyDirection);
  }
  void *devPtr;
  RETURN_CUDART_IF_ERROR(cudaGetSymbolAddress(&devPtr, symbol));
  char *src = static_cast<char *>(devPtr) + offset;
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
                                      size_t count, size_t offset,
                                      cudaMemcpyKind kind,
                                      cudaStream_t stream) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  if (kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyDeviceToDevice) {
    RETURN_CUDART(cudaErrorInvalidMemcpyDirection);
  }
  void *devPtr;
  RETURN_CUDART_IF_ERROR(cudaGetSymbolAddress(&devPtr, symbol));
  char *src = static_cast<char *>(devPtr) + offset;
  return cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                               size_t count, size_t offset,
                               cudaMemcpyKind kind) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToDevice) {
    RETURN_CUDART(cudaErrorInvalidMemcpyDirection);
  }
  void *devPtr;
  RETURN_CUDART_IF_ERROR(cudaGetSymbolAddress(&devPtr, symbol));
  char *dst = static_cast<char *>(devPtr) + offset;
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src,
                                    size_t count, size_t offset,
                                    cudaMemcpyKind kind, cudaStream_t stream) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToDevice) {
    RETURN_CUDART(cudaErrorInvalidMemcpyDirection);
  }
  void *devPtr;
  RETURN_CUDART_IF_ERROR(cudaGetSymbolAddress(&devPtr, symbol));
  char *dst = static_cast<char *>(devPtr) + offset;
  return cudaMemcpyAsync(dst, src, count, kind, stream);
}

static size_t getChannelBytes(CUarray_format format) {
  switch (format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
      //ABSL_FALLTHROUGH_INTENDED;
    case CU_AD_FORMAT_SIGNED_INT8:
      return 1;
    case CU_AD_FORMAT_UNSIGNED_INT16:
      //ABSL_FALLTHROUGH_INTENDED;
    case CU_AD_FORMAT_SIGNED_INT16:
      //ABSL_FALLTHROUGH_INTENDED;
    case CU_AD_FORMAT_HALF:
      return 2;
    case CU_AD_FORMAT_UNSIGNED_INT32:
      //ABSL_FALLTHROUGH_INTENDED;
    case CU_AD_FORMAT_SIGNED_INT32:
      //ABSL_FALLTHROUGH_INTENDED;
    case CU_AD_FORMAT_FLOAT:
      return 4;
  }
}

static CUresult getCudaMemcpy2DFromArrayParameters(
    CUDA_MEMCPY2D *parameters, void *dst, size_t dpitch, cudaArray_const_t src,
    size_t wOffset, size_t hOffset, size_t width, size_t height,
    cudaMemcpyKind kind) {
  CUarray cuSrc = (CUarray)src;
  CUDA_ARRAY_DESCRIPTOR arrayDescriptor;
  RETURN_IF_FAIL(cuArrayGetDescriptor(&arrayDescriptor, cuSrc));
  size_t elementSize =
      arrayDescriptor.NumChannels * getChannelBytes(arrayDescriptor.Format);

  // Common parameters.
  parameters->WidthInBytes = elementSize * width;
  parameters->Height = height;

  // Source parameters.
  parameters->srcMemoryType = CU_MEMORYTYPE_ARRAY;
  parameters->srcXInBytes = wOffset * elementSize;
  parameters->srcY = hOffset;
  parameters->srcArray = cuSrc;

  // Destination parameters.
  parameters->dstXInBytes = 0;
  parameters->dstY = 0;
  parameters->dstPitch = dpitch;
  switch (kind) {
    // device destination
    case cudaMemcpyDeviceToDevice:
    case cudaMemcpyHostToDevice:  // fallthrough
      parameters->dstMemoryType = CU_MEMORYTYPE_DEVICE;
      parameters->dstDevice = convertToCuDevicePtr(dst);
      break;
    // host destination
    case cudaMemcpyDeviceToHost:  // fallthrough
    case cudaMemcpyHostToHost:    // fallthrough
    case cudaMemcpyDefault:
      parameters->dstMemoryType = CU_MEMORYTYPE_HOST;
      parameters->dstHost = dst;
      break;
  }
  return CUDA_SUCCESS;
}

static CUresult getCudaMemcpy2DToArrayParameters(CUDA_MEMCPY2D *parameters,
                                                 cudaArray_t dst,
                                                 size_t wOffset, size_t hOffset,
                                                 const void *src, size_t spitch,
                                                 size_t width, size_t height,
                                                 cudaMemcpyKind kind) {
  CUarray cuDst = reinterpret_cast<CUarray>(dst);
  CUDA_ARRAY_DESCRIPTOR arrayDescriptor;
  RETURN_IF_FAIL(cuArrayGetDescriptor(&arrayDescriptor, cuDst));
  size_t elementSize =
      arrayDescriptor.NumChannels * getChannelBytes(arrayDescriptor.Format);

  // Common parameters.
  parameters->WidthInBytes = elementSize * width;
  parameters->Height = height;

  // Destination parameters.
  parameters->dstMemoryType = CU_MEMORYTYPE_ARRAY;
  parameters->dstXInBytes = wOffset * elementSize;
  parameters->dstY = hOffset;
  parameters->dstArray = cuDst;

  // Source parameters.
  parameters->srcXInBytes = 0;
  parameters->srcY = 0;
  parameters->srcPitch = spitch;
  switch (kind) {
    // device source
    case cudaMemcpyDeviceToHost:  // fallthrough
    case cudaMemcpyDeviceToDevice:
      parameters->srcMemoryType = CU_MEMORYTYPE_DEVICE;
      parameters->srcDevice = convertToCuDevicePtr(src);
      break;
    // host source
    case cudaMemcpyHostToDevice:  // fallthrough
    case cudaMemcpyHostToHost:    // fallthrough
    case cudaMemcpyDefault:
      parameters->srcMemoryType = CU_MEMORYTYPE_HOST;
      parameters->srcHost = src;
      break;
  }
  return CUDA_SUCCESS;
}

cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
                                  cudaArray_const_t src, size_t wOffset,
                                  size_t hOffset, size_t width, size_t height,
                                  cudaMemcpyKind kind) {
  CUDA_MEMCPY2D parameters;
  RETURN_CONVERTED_IF_FAIL(getCudaMemcpy2DFromArrayParameters(
      &parameters, dst, dpitch, src, wOffset, hOffset, width, height, kind));
  RETURN_CONVERTED(cuMemcpy2D(&parameters));
}

cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
                                       cudaArray_const_t src, size_t wOffset,
                                       size_t hOffset, size_t width,
                                       size_t height, cudaMemcpyKind kind,
                                       cudaStream_t stream) {
  CUDA_MEMCPY2D parameters;
  RETURN_CONVERTED_IF_FAIL(getCudaMemcpy2DFromArrayParameters(
      &parameters, dst, dpitch, src, wOffset, hOffset, width, height, kind));
  RETURN_CONVERTED(cuMemcpy2DAsync(&parameters, stream));
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                const void *src, size_t spitch, size_t width,
                                size_t height, cudaMemcpyKind kind) {
  CUDA_MEMCPY2D parameters;
  RETURN_CONVERTED_IF_FAIL(getCudaMemcpy2DToArrayParameters(
      &parameters, dst, wOffset, hOffset, src, spitch, width, height, kind));
  RETURN_CONVERTED(cuMemcpy2D(&parameters));
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset,
                                     size_t hOffset, const void *src,
                                     size_t spitch, size_t width, size_t height,
                                     cudaMemcpyKind kind, cudaStream_t stream) {
  CUDA_MEMCPY2D parameters;
  RETURN_CONVERTED_IF_FAIL(getCudaMemcpy2DToArrayParameters(
      &parameters, dst, wOffset, hOffset, src, spitch, width, height, kind));
  RETURN_CONVERTED(cuMemcpy2DAsync(&parameters, stream));
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  switch (kind) {
    case cudaMemcpyHostToHost:
      std::memcpy(dst, src, count);
      return cudaSuccess;
    case cudaMemcpyHostToDevice:
      RETURN_CONVERTED(cuMemcpyHtoD(convertToCuDevicePtr(dst), src, count));
    case cudaMemcpyDeviceToHost:
      RETURN_CONVERTED(cuMemcpyDtoH(dst, convertToCuDevicePtr(src), count));
    case cudaMemcpyDeviceToDevice:
      RETURN_CONVERTED(cuMemcpyDtoD(convertToCuDevicePtr(dst),
                                    convertToCuDevicePtr(src), count));
    default:
      RETURN_CONVERTED(cuMemcpy(convertToCuDevicePtr(dst),
                                convertToCuDevicePtr(src), count));
  }
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  switch (kind) {
    case cudaMemcpyHostToHost:
      // TODO: The host-to-host case is not asynchronous and it doesn't
      // pay attention to ordering in the stream. Is that OK?
      std::memcpy(dst, src, count);
      return cudaSuccess;
    case cudaMemcpyHostToDevice:
      RETURN_CONVERTED(
          cuMemcpyHtoDAsync(convertToCuDevicePtr(dst), src, count, stream));
    case cudaMemcpyDeviceToHost:
      RETURN_CONVERTED(
          cuMemcpyDtoHAsync(dst, convertToCuDevicePtr(src), count, stream));
    case cudaMemcpyDeviceToDevice:
      RETURN_CONVERTED(cuMemcpyDtoDAsync(
          convertToCuDevicePtr(dst), convertToCuDevicePtr(src), count, stream));
    default:
      RETURN_CONVERTED(cuMemcpyAsync(convertToCuDevicePtr(dst),
                                     convertToCuDevicePtr(src), count, stream));
  }
}

static CUDA_MEMCPY2D getCudaMemcpy2DParameters(void *dst, size_t dpitch,
                                               const void *src, size_t spitch,
                                               size_t width, size_t height,
                                               cudaMemcpyKind kind) {
  CUDA_MEMCPY2D parameters;
  switch (kind) {
    case cudaMemcpyDeviceToDevice:
      parameters.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      parameters.srcDevice = reinterpret_cast<CUdeviceptr>(src);
      parameters.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      parameters.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
      break;
    case cudaMemcpyDeviceToHost:
      parameters.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      parameters.srcDevice = reinterpret_cast<CUdeviceptr>(src);
      parameters.dstMemoryType = CU_MEMORYTYPE_HOST;
      parameters.dstHost = dst;
      break;
    case cudaMemcpyHostToDevice:
      parameters.srcMemoryType = CU_MEMORYTYPE_HOST;
      parameters.srcHost = src;
      parameters.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      parameters.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
      break;
    case cudaMemcpyHostToHost:
      parameters.srcMemoryType = CU_MEMORYTYPE_HOST;
      parameters.srcHost = src;
      parameters.dstMemoryType = CU_MEMORYTYPE_HOST;
      parameters.dstHost = dst;
      break;
    default:
      parameters.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
      parameters.srcHost = src;
      parameters.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
      parameters.dstHost = dst;
      break;
  }
  parameters.srcXInBytes = 0;
  parameters.srcY = 0;
  parameters.srcPitch = spitch;
  parameters.dstXInBytes = 0;
  parameters.dstY = 0;
  parameters.dstPitch = dpitch;
  parameters.WidthInBytes = width;
  parameters.Height = height;

  return parameters;
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                         size_t spitch, size_t width, size_t height,
                         cudaMemcpyKind kind) {
  CUDA_MEMCPY2D parameters =
      getCudaMemcpy2DParameters(dst, dpitch, src, spitch, width, height, kind);
  RETURN_CONVERTED(cuMemcpy2D(&parameters));
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                              size_t spitch, size_t width, size_t height,
                              cudaMemcpyKind kind, cudaStream_t stream) {
  CUDA_MEMCPY2D parameters =
      getCudaMemcpy2DParameters(dst, dpitch, src, spitch, width, height, kind);
  RETURN_CONVERTED(cuMemcpy2DAsync(&parameters, stream));
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuMemsetD8(convertToCuDevicePtr(devPtr), value, count));
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(
      cuMemsetD8Async(convertToCuDevicePtr(devPtr), value, count, stream));
}

// Peer device memory access

cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
                                    int peerDevice) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuDeviceCanAccessPeer(canAccessPeer,
                                         static_cast<CUdevice>(device),
                                         static_cast<CUdevice>(peerDevice)));
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
  ThreadInitializationState state = initializeThreadIfNeeded();
  RETURN_CONVERTED_IF_FAIL(state.resultCode);
  auto result = state.initializationState->GetContext(peerDevice);
  RETURN_CONVERTED_IF_FAIL(result.second);
  RETURN_CONVERTED(cuCtxDisablePeerAccess(result.first));
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
  ThreadInitializationState state = initializeThreadIfNeeded();
  RETURN_CONVERTED_IF_FAIL(state.resultCode);
  auto result = state.initializationState->GetContext(peerDevice);
  RETURN_CONVERTED_IF_FAIL(result.second);
  RETURN_CONVERTED(cuCtxEnablePeerAccess(result.first, flags));
}

// Texture reference management.

cudaError_t cudaBindTexture(size_t *offset, const textureReference *texref,
                            const void *devPtr,
                            const cudaChannelFormatDesc *desc, size_t size) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  CUtexref driverTexRef = getDriverTexRefFromHandle(texref);
  if (!driverTexRef) {
    return cudaErrorInvalidTexture;
  }

  CUarray_format format;
  switch (desc->f) {
    case cudaChannelFormatKindSigned:
      switch (desc->x) {
        case 8:
          format = CU_AD_FORMAT_SIGNED_INT8;
          break;
        case 16:
          format = CU_AD_FORMAT_SIGNED_INT16;
          break;
        case 32:
          format = CU_AD_FORMAT_SIGNED_INT32;
          break;
        default:
          return cudaErrorInvalidValue;
      }
      break;
    case cudaChannelFormatKindUnsigned:
      switch (desc->x) {
        case 8:
          format = CU_AD_FORMAT_UNSIGNED_INT8;
          break;
        case 16:
          format = CU_AD_FORMAT_UNSIGNED_INT16;
          break;
        case 32:
          format = CU_AD_FORMAT_UNSIGNED_INT32;
          break;
        default:
          return cudaErrorInvalidValue;
      }
      break;
    case cudaChannelFormatKindFloat:
      switch (desc->x) {
        case 16:
          format = CU_AD_FORMAT_HALF;
          break;
        case 32:
          format = CU_AD_FORMAT_FLOAT;
          break;
        default:
          return cudaErrorInvalidValue;
      }
      break;
    case cudaChannelFormatKindNone:
      return cudaErrorInvalidValue;
  }

  int channelCount = [desc]() {
    std::array<int, 4> sizes = {{desc->x, desc->y, desc->z, desc->w}};
    auto zero = std::find(sizes.begin(), sizes.end(), 0);
    return std::distance(sizes.begin(), zero);
  }();

  // The meaning of the numerical codes for these enums match in the driver
  // and runtime APIs, so static_cast does the right conversion.
  auto driverAddressMode = static_cast<CUaddress_mode>(texref->addressMode[0]);
  auto driverFilterMode = static_cast<CUfilter_mode>(texref->filterMode);
  auto driverMipmapFilterMode =
      static_cast<CUfilter_mode>(texref->mipmapFilterMode);

  RETURN_CONVERTED_IF_FAIL(
      cuTexRefSetAddressMode(driverTexRef, 0, driverAddressMode));
  RETURN_CONVERTED_IF_FAIL(
      cuTexRefSetFilterMode(driverTexRef, driverFilterMode));

  RETURN_CONVERTED_IF_FAIL(
      cuTexRefSetFormat(driverTexRef, format, channelCount));
  RETURN_CONVERTED_IF_FAIL(
      cuTexRefSetMaxAnisotropy(driverTexRef, texref->maxAnisotropy));
  RETURN_CONVERTED_IF_FAIL(
      cuTexRefSetMipmapFilterMode(driverTexRef, driverMipmapFilterMode));
  RETURN_CONVERTED_IF_FAIL(
      cuTexRefSetMipmapLevelBias(driverTexRef, texref->mipmapLevelBias));
  RETURN_CONVERTED_IF_FAIL(cuTexRefSetMipmapLevelClamp(
      driverTexRef, texref->minMipmapLevelClamp, texref->maxMipmapLevelClamp));
  unsigned int flags = texref->normalized ? CU_TRSF_NORMALIZED_COORDINATES
                                          : CU_TRSF_READ_AS_INTEGER;
  RETURN_CONVERTED_IF_FAIL(cuTexRefSetFlags(driverTexRef, flags));
  auto driverDevPtr = reinterpret_cast<CUdeviceptr>(devPtr);
  RETURN_CONVERTED(
      cuTexRefSetAddress(offset, driverTexRef, driverDevPtr, size));
}

cudaError_t cudaUnbindTexture(const textureReference *texref) {
  // Intentionally left as a no-op because this function has no effect at the
  // driver level.
  return cudaSuccess;
}

cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
                                            cudaChannelFormatKind f) {
  cudaChannelFormatDesc result;
  result.x = x;
  result.y = y;
  result.z = z;
  result.w = w;
  result.f = f;
  return result;
}

// Texture object management.

static CUDA_RESOURCE_DESC convertToCuResourceDesc(
    const cudaResourceDesc &resDesc) {
  CUDA_RESOURCE_DESC cuResDesc;
  std::memset(&cuResDesc, 0, sizeof(cuResDesc));

  cuResDesc.resType = static_cast<CUresourcetype>(resDesc.resType);
  cuResDesc.flags = 0u;

  switch (resDesc.resType) {
    case cudaResourceTypeArray:
      cuResDesc.res.array.hArray =
          reinterpret_cast<CUarray>(resDesc.res.array.array);
      break;
    case cudaResourceTypeMipmappedArray:
      cuResDesc.res.mipmap.hMipmappedArray =
          reinterpret_cast<CUmipmappedArray>(resDesc.res.mipmap.mipmap);
      break;
    case cudaResourceTypeLinear:
      cuResDesc.res.linear.devPtr =
          convertToCuDevicePtr(resDesc.res.linear.devPtr);
      cuResDesc.res.linear.format = getCuArrayFormat(resDesc.res.linear.desc);
      cuResDesc.res.linear.numChannels =
          getNumChannels(resDesc.res.linear.desc);
      cuResDesc.res.linear.sizeInBytes = resDesc.res.linear.sizeInBytes;
      break;
    case cudaResourceTypePitch2D:
      cuResDesc.res.pitch2D.devPtr =
          convertToCuDevicePtr(resDesc.res.pitch2D.devPtr);
      cuResDesc.res.pitch2D.format = getCuArrayFormat(resDesc.res.linear.desc);
      cuResDesc.res.pitch2D.numChannels =
          getNumChannels(resDesc.res.linear.desc);
      cuResDesc.res.pitch2D.width = resDesc.res.pitch2D.width;
      cuResDesc.res.pitch2D.height = resDesc.res.pitch2D.height;
      cuResDesc.res.pitch2D.pitchInBytes = resDesc.res.pitch2D.pitchInBytes;
      break;
  }

  return cuResDesc;
}

static CUDA_TEXTURE_DESC convertToCuTextureDesc(
    const cudaTextureDesc &texDesc) {
  CUDA_TEXTURE_DESC cuTexDesc;
  std::memset(&cuTexDesc, 0, sizeof(cuTexDesc));
  for (int i = 0; i < 3; ++i) {
    cuTexDesc.addressMode[i] =
        static_cast<CUaddress_mode>(texDesc.addressMode[i]);
  }
  cuTexDesc.filterMode = static_cast<CUfilter_mode>(texDesc.filterMode);
  cuTexDesc.flags =
      texDesc.normalizedCoords ? CU_TRSF_NORMALIZED_COORDINATES : 0u;
  cuTexDesc.maxAnisotropy = texDesc.maxAnisotropy;
  cuTexDesc.maxMipmapLevelClamp = texDesc.maxMipmapLevelClamp;
  cuTexDesc.minMipmapLevelClamp = texDesc.minMipmapLevelClamp;
  cuTexDesc.mipmapFilterMode =
      static_cast<CUfilter_mode>(texDesc.mipmapFilterMode);
  cuTexDesc.mipmapLevelBias = texDesc.mipmapLevelBias;
  return cuTexDesc;
}

static CUDA_RESOURCE_VIEW_DESC convertToCuResourceViewDesc(
    const cudaResourceViewDesc &resViewDesc) {
  CUDA_RESOURCE_VIEW_DESC cuResViewDesc;
  std::memset(&cuResViewDesc, 0, sizeof(cuResViewDesc));
  cuResViewDesc.depth = resViewDesc.depth;
  cuResViewDesc.firstLayer = resViewDesc.firstLayer;
  cuResViewDesc.firstMipmapLevel = resViewDesc.firstMipmapLevel;
  cuResViewDesc.format = static_cast<CUresourceViewFormat>(resViewDesc.format);
  cuResViewDesc.height = resViewDesc.height;
  cuResViewDesc.lastLayer = resViewDesc.lastLayer;
  cuResViewDesc.lastMipmapLevel = resViewDesc.lastMipmapLevel;
  cuResViewDesc.width = resViewDesc.width;
  return cuResViewDesc;
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTextObject,
                                    const cudaResourceDesc *pResDesc,
                                    const cudaTextureDesc *pTexDesc,
                                    const cudaResourceViewDesc *pResViewDesc) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  CUDA_RESOURCE_DESC cuResDesc = convertToCuResourceDesc(*pResDesc);
  CUDA_TEXTURE_DESC cuTexDesc = convertToCuTextureDesc(*pTexDesc);
  if (!pResViewDesc) {
    RETURN_CONVERTED(
        cuTexObjectCreate(pTextObject, &cuResDesc, &cuTexDesc, nullptr));
  } else {
    CUDA_RESOURCE_VIEW_DESC cuResViewDesc =
        convertToCuResourceViewDesc(*pResViewDesc);
    RETURN_CONVERTED(
        cuTexObjectCreate(pTextObject, &cuResDesc, &cuTexDesc, &cuResViewDesc));
  }
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
  RETURN_CONVERTED_IF_FAIL(initializeThreadIfNeeded().resultCode);
  RETURN_CONVERTED(cuTexObjectDestroy(texObject));
}

cudaError_t cudaGetTextureObjectResourceDesc(
    cudaResourceDesc * /*pResDesc*/, cudaTextureObject_t /*texObject*/) {
  // TODO: Implement.
  return cudaErrorUnknown;
}

cudaError_t cudaGetTextureObjectResourceViewDesc(
    cudaResourceViewDesc * /*pResViewDesc*/,
    cudaTextureObject_t /*texObject*/) {
  // TODO: Implement.
  return cudaErrorUnknown;
}

cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc * /*pTexDesc*/,
                                            cudaTextureObject_t /*texObject*/) {
  // TODO: Implement.
  return cudaErrorUnknown;
}
