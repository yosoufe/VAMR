#include "cuda_lib.cuh"
#include "cuda_lib.hpp"

__global__
void cuda::example_kernel()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}


void cuda::kernel_caller(size_t n_blocks, size_t n_threads)
{
    cuda::example_kernel<<<n_blocks, n_threads>>>();
    cudaDeviceSynchronize();
}
