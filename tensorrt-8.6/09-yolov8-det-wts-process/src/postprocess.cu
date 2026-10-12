#include "postprocess.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>


namespace
{
#define POSTPROCESS_CUDA_CHECK(call)                                                     \
    do                                                                                  \
    {                                                                                   \
        cudaError_t const error__ = (call);                                              \
        if (error__ != cudaSuccess)                                                      \
        {                                                                               \
            throw std::runtime_error(                                                    \
                std::string("CUDA error: ") + cudaGetErrorString(error__) +            \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));                  \
        }                                                                               \
    } while (false)

constexpr int32_t kDecodedWidth = 6;
constexpr int32_t kNmsThreads = 64;



}

