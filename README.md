# Simple GPU runtime support library based on driver API 

This library allows running CUDA applications using only the driver API
(https://docs.nvidia.com/cuda/archive/9.0/cuda-driver-api/index.html).  It's
functional enough to run simple CUDA applications, but is not tested beyond
that.

**NOTE**: Google is not planning to make any further changes to this library. It's a
proof-of-concept implementation released in hope that it will be useful on
platforms where the CUDA runtime library is not available.

## Limitations

This library only works for CUDA code compiled with CUDA-9.0 or older. More
recent CUDA versions use slightly different API to set up and launch kernels and
will need to implement a handful of additional functions.

The library has only been tested on Linux.

## Building

Prerequisites: clang-7.0.0 or newer. 

```shell
$ # git clone <...>/gpu-runtime.git
$ cd gpu-runtime
$ git submodule update --init --recursive
$ mkdir build
$ cd build
$ cmake cmake -DCUDA_ROOT=/path/to/cuda-9.0 -DCMAKE_CXX_COMPILER=/path/to/recent/clang++ ../
$ make -j 8
$ make test
```

## Using

Link your application with `gpu-runtime.o` instead of `-lcudart`.

