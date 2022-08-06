#pragma once
#include "cuda_types.hpp"
#include <thrust/device_ptr.h>

namespace cuda
{

    template <typename T>
    thrust::device_ptr<T> thrust_ptr_begin(const CuMatrix<T> &input) { return thrust::device_pointer_cast<T>(input.data()); };

    template <typename T>
    thrust::device_ptr<T> thrust_ptr_end(const CuMatrix<T> &input) { return thrust_ptr_begin<T>(input) + input.n_elements(); };
}