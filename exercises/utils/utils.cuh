#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <thrust/tuple.h>

/**
 * @brief      CUDA safe call.
 *
 * @param[in]  err          The error
 * @param[in]  msg          The message
 * @param[in]  file_name    The file name
 * @param[in]  line_number  The line number
 */
inline void _safe_cuda_call(cudaError err, const char *file_name, const int line_number)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "\nFile: %s:%d\nReason: %s\n\n", file_name, line_number, cudaGetErrorString(err));
        throw std::runtime_error("cuda call issue!");
    }
}

inline void _cuda_last_error(const char *file_name, const int line_number)
{
    auto cuda_erro = cudaGetLastError();
    _safe_cuda_call(cuda_erro, file_name, line_number);
}

// Safe call
#define CSC(call) _safe_cuda_call((call), __FILE__, __LINE__)
// CUDA LAST Error
#define CLE() _cuda_last_error(__FILE__, __LINE__)

inline void _cudnn_error(cudnnStatus_t err, const char *file_name, const int line_number)
{
    if (err != CUDNN_STATUS_SUCCESS)
    {
        fprintf(stderr, "\nCUDNN Error:  File: %s:%d\nReason: %s\n\n", file_name, line_number, cudnnGetErrorString(err));
        std::exit(1);
    }
}

#define CUDNN_CALL(call) _cudnn_error((call), __FILE__, __LINE__)

__host__ __device__ int
get_index_rowwise(int row, int col, int n_cols, int stride = 1);

__host__ __device__ int
get_index_colwise(int row, int col, int n_rows, int stride = 1);

__host__ __device__ thrust::tuple<int, int>
get_2d_index_rowwise(int index_1d, int n_cols);

__host__ __device__ thrust::tuple<int, int>
get_2d_index_colwise(int index_1d, int n_rows);

template <typename T>
__global__ void print_cuda_eigen(T *data, int cols, int rows)
{
    printf("printing in cuda kernel:\n");
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            int idx = get_index_colwise(row, col, rows, 1);
            printf("%d: ", idx);
            printf("%f ,", float(data[idx]));
        }
        printf("\n");
    }
}