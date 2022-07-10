#include "utils.cuh"
#include "utils.hpp"


__device__ int
get_index_rowwise(int row, int col, int n_cols, int stride)
{
    // printf("row: %d, col: %d, n_cols: %d\n", row, col, n_cols);
    return (col + n_cols * row) * stride;
}

__device__ int
get_index_colwise(int row, int col, int n_rows, int stride)
{
    // printf("row: %d, col: %d, n_rows: %d\n", row, col, n_rows);
    return (row + n_rows * col) * stride;
}