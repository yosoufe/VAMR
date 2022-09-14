#pragma once

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/execution_policy.h>
#include "cuda_types.hpp"
#include "cuda_types.cuh"

namespace cuda
{
    template<typename T>
    thrust::device_ptr<int> create_indices(const cuda::CuMatrix<T> &input)
    {
        thrust::device_ptr<int> d_output = thrust::device_malloc<int>(input.n_elements());
        thrust::sequence(thrust::device, d_output, d_output + input.n_elements());
        return d_output;
    }
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
