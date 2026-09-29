// Error-checking macro for CUDA runtime calls.
#ifndef CONNECT4_CUDA_CHECK_CUH_
#define CONNECT4_CUDA_CHECK_CUH_

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                   cudaGetErrorString(err_), __FILE__, __LINE__);          \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

#endif  // CONNECT4_CUDA_CHECK_CUH_
