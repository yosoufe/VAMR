#include "utils.cuh"
#include "utils.hpp"

__host__ __device__ int
get_index_rowwise(int row, int col, int n_cols, int stride)
{
    // printf("row: %d, col: %d, n_cols: %d\n", row, col, n_cols);
    return (col + n_cols * row) * stride;
}

__host__ __device__ int
get_index_colwise(int row, int col, int n_rows, int stride)
{
    // printf("row: %d, col: %d, n_rows: %d\n", row, col, n_rows);
    return (row + n_rows * col) * stride;
}

__host__ __device__ thrust::tuple<int, int>
get_2d_index_rowwise(int index_1d, int n_cols)
{
    int row = index_1d / n_cols;
    int col = index_1d % n_cols;
    return thrust::tuple<int, int>(row, col);
}
__host__ __device__ thrust::tuple<int, int>
get_2d_index_colwise(int index_1d, int n_rows)
{
    int row = index_1d % n_rows;
    int col = index_1d / n_rows;
    return thrust::tuple<int, int>(row, col);
}