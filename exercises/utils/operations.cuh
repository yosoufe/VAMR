#pragma once

#include <thrust/device_ptr.h>
#include "cuda_types.hpp"

namespace cuda
{
    thrust::device_ptr<int> create_indices(const cuda::CuMatrixD &input);
}
