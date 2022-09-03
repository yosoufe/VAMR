#pragma once

#include <thrust/device_ptr.h>
#include "cuda_types.hpp"
#include "cuda_types.cuh"

namespace cuda
{
    thrust::device_ptr<int> create_indices(const cuda::CuMatrixD &input);
}

__global__ void
sum_kernel(
    double *output,
    double *input,
    int num_items);

__global__ void
difference_kernel(
    double* output,
    double* input_1,
    double* input_2,
    int num_items
);
