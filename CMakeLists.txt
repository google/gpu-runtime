# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.13)

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "clang")
endif()
list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)

project(gpu-runtime LANGUAGES CXX)

if (NOT CUDA_ROOT)
  set(CUDA_ROOT /usr/local/cuda)
endif()

# Pick the C++ standard to compile with.
# Abseil currently supports C++11, C++14, and C++17.
set(CMAKE_CXX_STANDARD 11)
add_subdirectory(lib/abseil-cpp)
add_subdirectory(lib/googletest)

list(APPEND CUDA_FLAGS "-x" "cuda")
list(APPEND CUDA_FLAGS "--cuda-path=${CUDA_ROOT}")
list(APPEND CUDA_FLAGS "--cuda-gpu-arch=sm_70")
include_directories(${CUDA_ROOT}/include .)

# Import some CUDA lbiraries
add_library(libcuda SHARED IMPORTED)
set_property(TARGET libcuda PROPERTY IMPORTED_LOCATION
  ${CUDA_ROOT}/lib64/stubs/libcuda.so)
find_library(LIBCULIBOS culibos PATHS ${CUDA_ROOT}/lib64)
find_library(LIBCUBLAS cublas_static PATHS ${CUDA_ROOT}/lib64)

# Runtime support library itself
add_library(gpu-runtime gpu-runtime.cc)
target_include_directories(gpu-runtime PUBLIC ${CUDA_ROOT}/include)
target_link_libraries(gpu-runtime absl::base absl::synchronization absl::strings libcuda)

add_library(culibos STATIC IMPORTED)
set_target_properties(culibos PROPERTIES
  IMPORTED_LOCATION ${LIBCULIBOS})

add_library(cublas STATIC IMPORTED)
set_target_properties(cublas PROPERTIES
  IMPORTED_LOCATION ${LIBCUBLAS}
  INTERFACE_LINK_LIBRARIES "culibos;gpu-runtime;${CMAKE_DL_LIBS}")

# Few simple examples
add_executable(saxpy examples/saxpy.cu.cc)
target_compile_options(saxpy PRIVATE "${CUDA_FLAGS}")
target_link_libraries(saxpy gpu-runtime)

add_executable(quickstart42 examples/quickstart42.cu.cc)
target_compile_options(quickstart42 PRIVATE ${CUDA_FLAGS})
target_link_libraries(quickstart42 gpu-runtime)

# Some tests
add_custom_target(all-tests COMMENT "Build all tests")

macro(make_test_name TestName TestSourcePath)
  string(REGEX REPLACE ".*/" "" _tmp ${TestSourcePath})
  string(REPLACE "/" "-" _tmp ${_tmp})
  string(REGEX REPLACE "\\..*$" "" _tmp ${_tmp})
  set(${TestName} ${_tmp})
endmacro()

macro(make_test TestSourcePath)
  make_test_name(_TestName ${TestSourcePath})
  add_executable(${_TestName} ${TestSourcePath})
  if(${TestSourcePath} MATCHES ".*\.cu(\.cc)?$")
    target_compile_options(${_TestName} PRIVATE ${CUDA_FLAGS})
  endif()
  target_link_libraries(${_TestName} ${EXTRA_DEPS_${_TestName}}
    gpu-runtime gmock gtest_main absl::strings
    ${CMAKE_DL_LIBS})
  add_dependencies(all-tests ${_TestName})
  add_test(NAME "run-${_TestName}" COMMAND ${_TestName})
endmacro()

macro(make_all_tests)
  file(GLOB_RECURSE _FileList "tests/*.cc")
  foreach(_File IN LISTS _FileList)
    make_test(${_File})
  endforeach()
endmacro()

# Augment some tests with additional dependencies.
add_library(device_code tests/device_code.cu)
set_source_files_properties(tests/device_code.cu PROPERTIES LANGUAGE CXX)
target_compile_options(device_code PRIVATE ${CUDA_FLAGS})
target_link_libraries(device_code gpu-runtime)

set(EXTRA_DEPS_deprecated_execution_control_test device_code)
set(EXTRA_DEPS_execution_control_test device_code)
set(EXTRA_DEPS_cublas_static_test cublas)

make_all_tests()
enable_testing()
